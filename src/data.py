import numpy as np
import pandas as pd

def generate_transactions(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=8.0, sigma=1.0, size=n), 100, 250000)
    hour = rng.integers(0,24,n)
    account_age = np.clip(rng.exponential(900,n).astype(int),0,5000)
    velocity = np.clip(rng.poisson(1.8,n),0,15)
    failed = np.clip(rng.poisson(.35,n),0,8)
    distance = np.clip(rng.exponential(90,n),0,3000)
    channel = rng.choice(["Mobile","Web","POS"],n,p=[.48,.37,.15])
    international = rng.binomial(1,.12,n)
    merchant_risk = rng.choice([0,1,2],n,p=[.55,.32,.13])
    device_trust = rng.choice([0,1,2],n,p=[.12,.28,.60])

    # A transparent synthetic label-generation rule with a little randomness.
    z = (-5.0
         + .000012*amount
         + .105*velocity
         + .22*failed
         + .0011*distance
         + .75*international
         + .62*merchant_risk
         + .65*(device_trust==0)
         + .42*(hour<=4)
         + .55*(account_age<30))
    prob = 1/(1+np.exp(-z))
    fraud = rng.binomial(1,prob)
    days = rng.integers(0,90,n)
    ids = [f"TXN-{i+1:06d}" for i in range(n)]
    df = pd.DataFrame({
        "transaction_id":ids,"day":pd.to_datetime("2026-01-01")+pd.to_timedelta(days,unit="D"),
        "amount":amount.round(2),"hour":hour,"account_age_days":account_age,
        "txn_velocity_1h":velocity,"failed_attempts":failed,"distance_km":distance.round(1),
        "channel":channel,"international":international,"merchant_risk":merchant_risk,
        "device_trust":device_trust,"is_fraud":fraud
    })
    return df
