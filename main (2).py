import json, os, re, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

LEDGER_URL = os.environ.get("LEDGER_URL", "").strip()
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "").strip()
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
TZ = "Asia/Kolkata"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state = {"df": None, "rates": None}
_lock = threading.Lock()


# ---------------------------------------------------------------- data loading
def _with_params(url, **upd):
    p = urlsplit(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q.update({k: str(v) for k, v in upd.items()})
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))


def _get(url, tries=4):
    last = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa
            last = e
    raise last


def load_data():
    root = _get(LEDGER_URL)
    links = root["links"]
    rates = _get(links["rates"])["usd_per_unit"]
    first = _get(links["orders"])
    pages = int(first.get("pages", 1))
    rows = list(first["orders"])
    if pages > 1:
        urls = [_with_params(links["orders"], page=n) for n in range(2, pages + 1)]
        with ThreadPoolExecutor(max_workers=16) as ex:
            for d in ex.map(_get, urls):
                rows.extend(d["orders"])
    return build_df(rows, rates), rates


def build_df(rows, rates):
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")
    # an order id can appear more than once: latest updated_at wins
    df = df.sort_values("updated_at", kind="stable").drop_duplicates("id", keep="last")
    df["created_at"] = df["created_at"].dt.tz_convert(TZ)
    df["updated_at"] = df["updated_at"].dt.tz_convert(TZ)
    df["amount_usd"] = df["amount"].astype(float) * df["currency"].map(rates).astype(float)
    df["unit_price_usd"] = df["unit_price"].astype(float) * df["currency"].map(rates).astype(float)
    df["month"] = df["created_at"].dt.strftime("%Y-%m")
    df["date"] = df["created_at"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def get_df():
    if _state["df"] is None:
        with _lock:
            if _state["df"] is None:
                _state["df"], _state["rates"] = load_data()
    return _state["df"]


@app.on_event("startup")
def _warm():
    if LEDGER_URL:
        threading.Thread(target=get_df, daemon=True).start()


# ---------------------------------------------------------------- LLM -> pandas
SYSTEM = """You translate questions about an order ledger into pandas code.
A pandas DataFrame `df` is already loaded (numpy as np, pandas as pd available). It has ONE row per order
(duplicates already removed, latest version kept). Columns:
 id, customer (e.g. 'C1082'), region (North/South/East/West/Central), product, qty (units),
 unit_price, amount (qty*unit_price in original currency), currency (USD/EUR/INR),
 status ('paid','refunded','void', possibly others),
 created_at, updated_at (tz-aware pandas Timestamps in Asia/Kolkata),
 amount_usd (amount converted to USD), unit_price_usd,
 month (string 'YYYY-MM' from created_at, IST), date (string 'YYYY-MM-DD' from created_at, IST).
Business rules:
 - REVENUE = sum of amount_usd over rows with status == 'paid' only. Refunded/void orders are never revenue.
 - REFUNDS = rows with status == 'refunded'; refund value = amount_usd of those rows.
 - Dates/months always refer to created_at in Asia/Kolkata (use the `month` / `date` columns).
 - All money is in USD, use amount_usd. Do not round inside the code unless asked.
 - Match region/product/customer text case-insensitively against the data's own spelling
   (e.g. df['region'].str.lower()=='north'). Product names in the answer must be the exact spelling from df.
 - 'orders' means order rows; 'units' means sum of qty; 'customers' means distinct customer ids.
 - For ties in rankings, break ties alphabetically by name.
Write python that ends by assigning the final answer to a variable named `result`.
`result` must be a plain number, string, boolean, list or dict (for example a count, a USD total,
a product name, or a list of names). Output ONLY the code, no explanation, no markdown fences."""


def ask_llm(question, extra=""):
    r = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={"model": MODEL, "temperature": 0,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"Question: {question}\n{extra}"}]},
        timeout=11,
    )
    r.raise_for_status()
    code = r.json()["choices"][0]["message"]["content"].strip()
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code, flags=re.S).strip()
    return code


SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ("len", "sum", "min", "max", "abs", "round", "sorted", "list", "dict", "set",
                           "str", "int", "float", "bool", "range", "enumerate", "zip", "any", "all",
                           "tuple", "isinstance", "map", "filter", "reversed")}


def run_code(code, df):
    if re.search(r"\b(import|open|exec|eval|__|os\.|sys\.|subprocess)\b", code.replace("import numpy", "").replace("import pandas", "")):
        raise ValueError("unsafe code")
    ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
    return ns["result"]


def clean(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return round(x, 2)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Series):
        return [clean(v) for v in x.tolist()] if x.index.equals(pd.RangeIndex(len(x))) else {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return clean(x.to_dict("records"))
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    return x


def answer_question(q):
    df = get_df()
    err = ""
    for attempt in range(2):
        try:
            code = ask_llm(q, err)
            return clean(run_code(code, df))
        except Exception as e:  # noqa
            err = f"Your previous code failed with: {e!r}. Fix it."
    raise RuntimeError("could not answer")


@app.post("/")
@app.post("/ask")
def ask(body: dict):
    q = str(body.get("question", "")).strip()
    try:
        return {"answer": answer_question(q)}
    except Exception as e:  # noqa
        return {"answer": None, "error": str(e)}


@app.get("/")
def health():
    df = _state["df"]
    return {"status": "ok", "orders_loaded": None if df is None else len(df)}


@app.get("/reload")
def reload():
    _state["df"] = None
    return {"orders_loaded": len(get_df())}
