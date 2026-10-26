"""Technical indicators for the AMV-TrendBreak algorithmic trading strategy.

Includes high-performance vectorized implementations of:
- Exponential Moving Average (EMA: 9, 21, 50, 200)
- Average True Range (ATR: 14) and Normalized ATR
- Keltner Channels (EMA 20 +/- 1.8 * ATR 14)
- Average Directional Index (ADX: 14, +DI, -DI)
- Choppiness Index (CHOP: 14)
- Relative Strength Index (RSI: 14)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence, Union
import numpy as np
import pandas as pd


class KeltnerChannels(NamedTuple):
    """Keltner Channels upper, middle, and lower bands."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series


class ADXResult(NamedTuple):
    """ADX and Directional Indicators."""
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


class BollingerBands(NamedTuple):
    """Bollinger Bands upper, middle, and lower bands."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series



@dataclass
class IndicatorSnapshot:
    """Snapshot of technical indicator values for a single bar."""
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float
    atr_14: float
    natr_14: float
    kc_upper: float
    kc_middle: float
    kc_lower: float
    adx_14: float
    plus_di: float
    minus_di: float
    chop_14: float
    rsi_14: float
    volume_sma_20: float
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bbw: float = 0.0
    bbw_sma: float = 0.0


def calculate_ema(data: Union[pd.Series, Sequence[float], np.ndarray], period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA) with Wilder/Standard alpha.

    Formula:
        alpha = 2 / (period + 1)
        EMA_t = alpha * Price_t + (1 - alpha) * EMA_{t-1}
    """
    if not isinstance(data, pd.Series):
        series = pd.Series(data, dtype=float)
    else:
        series = data.astype(float)

    if len(series) == 0:
        return pd.Series(dtype=float)

    return series.ewm(span=period, adjust=False).mean()


def calculate_true_range(
    high: Union[pd.Series, Sequence[float], np.ndarray],
    low: Union[pd.Series, Sequence[float], np.ndarray],
    close: Union[pd.Series, Sequence[float], np.ndarray],
) -> pd.Series:
    """Calculate True Range (TR).

    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    """
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    c_prev = c.shift(1)

    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # First bar TR is simply High - Low
    if len(tr) > 0:
        tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr


def calculate_atr(
    high: Union[pd.Series, Sequence[float], np.ndarray],
    low: Union[pd.Series, Sequence[float], np.ndarray],
    close: Union[pd.Series, Sequence[float], np.ndarray],
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder's exponential smoothing."""
    tr = calculate_true_range(high, low, close)
    if len(tr) == 0:
        return pd.Series(dtype=float)

    # Wilder's smoothing corresponds to ewm with alpha = 1 / period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def calculate_keltner_channels(
    high: Union[pd.Series, Sequence[float], np.ndarray],
    low: Union[pd.Series, Sequence[float], np.ndarray],
    close: Union[pd.Series, Sequence[float], np.ndarray],
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 1.8,
) -> KeltnerChannels:
    """Calculate Keltner Channels.

    Middle Line = EMA(close, 20)
    Upper Channel = Middle Line + multiplier * ATR(14)
    Lower Channel = Middle Line - multiplier * ATR(14)
    """
    c = pd.Series(close, dtype=float)
    mid = calculate_ema(c, period=ema_period)
    atr = calculate_atr(high, low, c, period=atr_period)

    upper = mid + (multiplier * atr)
    lower = mid - (multiplier * atr)

    return KeltnerChannels(upper=upper, middle=mid, lower=lower)


def calculate_adx(
    high: Union[pd.Series, Sequence[float], np.ndarray],
    low: Union[pd.Series, Sequence[float], np.ndarray],
    close: Union[pd.Series, Sequence[float], np.ndarray],
    period: int = 14,
) -> ADXResult:
    """Calculate Average Directional Index (ADX) and Directional Indicators (+DI, -DI)."""
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)

    delta_h = h.diff()
    delta_l = -l.diff()

    # Directional Movement
    plus_dm = np.where((delta_h > delta_l) & (delta_h > 0.0), delta_h, 0.0)
    minus_dm = np.where((delta_l > delta_h) & (delta_l > 0.0), delta_l, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=h.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=l.index, dtype=float)

    # Wilder smoothing (alpha = 1 / period)
    smoothed_plus_dm = plus_dm_s.ewm(alpha=1.0 / period, adjust=False).mean()
    smoothed_minus_dm = minus_dm_s.ewm(alpha=1.0 / period, adjust=False).mean()
    atr = calculate_atr(h, l, c, period=period)

    # Avoid division by zero
    safe_atr = atr.replace(0.0, np.nan)
    plus_di = 100.0 * (smoothed_plus_dm / safe_atr).fillna(0.0)
    minus_di = 100.0 * (smoothed_minus_dm / safe_atr).fillna(0.0)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    # Directional Index (DX)
    dx = 100.0 * (di_diff / di_sum.replace(0.0, np.nan)).fillna(0.0)

    # ADX = Wilder smoothed DX
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    return ADXResult(adx=adx, plus_di=plus_di, minus_di=minus_di)


def calculate_choppiness(
    high: Union[pd.Series, Sequence[float], np.ndarray],
    low: Union[pd.Series, Sequence[float], np.ndarray],
    close: Union[pd.Series, Sequence[float], np.ndarray],
    period: int = 14,
) -> pd.Series:
    """Calculate Choppiness Index (CHOP).

    Formula:
        CHOP = 100 * LOG10( SUM(TR, period) / (MaxHigh(period) - MinLow(period)) ) / LOG10(period)
    Values:
        > 61.8 : Choppy / consolidating market (suppress breakout entries)
        < 38.2 : Strong explosive trend
    """
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)

    tr = calculate_true_range(h, l, c)
    sum_tr = tr.rolling(window=period, min_periods=1).sum()
    max_h = h.rolling(window=period, min_periods=1).max()
    min_l = l.rolling(window=period, min_periods=1).min()

    price_range = max_h - min_l
    safe_range = price_range.replace(0.0, np.nan)

    ratio = (sum_tr / safe_range).fillna(1.0)
    # Clip ratio to avoid non-positive log
    ratio = ratio.clip(lower=1e-9)

    log_n = np.log10(float(period))
    chop = 100.0 * (np.log10(ratio) / log_n)

    # Bound CHOP to [0, 100]
    chop = chop.clip(lower=0.0, upper=100.0)
    return chop


def calculate_rsi(
    close: Union[pd.Series, Sequence[float], np.ndarray],
    period: int = 14,
) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    Uses Wilder's smoothing on price gains and losses.
    """
    c = pd.Series(close, dtype=float)
    delta = c.diff()

    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    gain_s = pd.Series(gain, index=c.index, dtype=float)
    loss_s = pd.Series(loss, index=c.index, dtype=float)

    avg_gain = gain_s.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss_s.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = (avg_gain / avg_loss.replace(0.0, np.nan)).fillna(100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Special handling: if loss is 0 and gain > 0 -> RSI = 100; if both 0 -> 50
    zero_loss = (avg_loss == 0.0)
    zero_gain = (avg_gain == 0.0)
    rsi = rsi.mask(zero_loss & ~zero_gain, 100.0)
    rsi = rsi.mask(zero_loss & zero_gain, 50.0)

    return rsi.clip(lower=0.0, upper=100.0)


def calculate_bollinger_bands(
    close: Union[pd.Series, Sequence[float], np.ndarray],
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerBands:
    """Calculate Bollinger Bands.

    Middle Band = SMA(close, period)
    Upper Band = Middle Band + num_std * StdDev(close, period)
    Lower Band = Middle Band - num_std * StdDev(close, period)
    """
    if not isinstance(close, pd.Series):
        c = pd.Series(close, dtype=float)
    else:
        c = close.astype(float)

    if len(c) == 0:
        empty = pd.Series(dtype=float)
        return BollingerBands(upper=empty, middle=empty, lower=empty)

    middle = c.rolling(window=period, min_periods=1).mean()
    std = c.rolling(window=period, min_periods=1).std(ddof=0).fillna(0.0)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return BollingerBands(upper=upper, middle=middle, lower=lower)


def calculate_bbw(
    close: Union[pd.Series, Sequence[float], np.ndarray],
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Calculate Bollinger Band Width (BBW).

    Formula:
        BBW = (Upper Band - Lower Band) / Middle Band
    """
    bb = calculate_bollinger_bands(close, period=period, num_std=num_std)
    safe_mid = bb.middle.replace(0.0, np.nan)
    bbw = ((bb.upper - bb.lower) / safe_mid).fillna(0.0)
    return bbw


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full set of technical indicators on an OHLCV DataFrame.

    Requires columns: 'open', 'high', 'low', 'close', and optionally 'volume'.
    Returns a new DataFrame enriched with indicator columns:
    - ema_9, ema_21, ema_50, ema_200
    - atr_14, natr_14
    - kc_upper, kc_middle, kc_lower
    - adx_14, plus_di, minus_di
    - chop_14
    - rsi_14
    - volume_sma_20
    - bb_upper, bb_middle, bb_lower
    - bbw, bbw_sma
    """
    # Standardize column names to lowercase
    data = df.copy()
    col_map = {col: str(col).lower() for col in data.columns}
    data = data.rename(columns=col_map)

    required = ['high', 'low', 'close']
    for req in required:
        if req not in data.columns:
            raise ValueError(f"Missing required column '{req}' in DataFrame")

    high = data['high'].astype(float)
    low = data['low'].astype(float)
    close = data['close'].astype(float)

    # 1. EMAs
    data['ema_9'] = calculate_ema(close, 9)
    data['ema_21'] = calculate_ema(close, 21)
    data['ema_50'] = calculate_ema(close, 50)
    data['ema_200'] = calculate_ema(close, 200)

    # 2. ATR & NATR
    atr_14 = calculate_atr(high, low, close, 14)
    data['atr_14'] = atr_14
    data['natr_14'] = (atr_14 / close) * 100.0

    # 3. Keltner Channels (20 and 14, 1.8 ATR)
    kc = calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, multiplier=1.8)
    data['kc_upper'] = kc.upper
    data['kc_middle'] = kc.middle
    data['kc_lower'] = kc.lower

    kc14 = calculate_keltner_channels(high, low, close, ema_period=14, atr_period=14, multiplier=1.8)
    data['kc_14_upper'] = kc14.upper
    data['kc_14_middle'] = kc14.middle
    data['kc_14_lower'] = kc14.lower

    # 4. ADX (14)
    adx_res = calculate_adx(high, low, close, 14)
    data['adx_14'] = adx_res.adx
    data['plus_di'] = adx_res.plus_di
    data['minus_di'] = adx_res.minus_di

    # 5. Choppiness Index (14)
    data['chop_14'] = calculate_choppiness(high, low, close, 14)

    # 6. RSI (14)
    data['rsi_14'] = calculate_rsi(close, 14)

    # 7. Volume SMA (20)
    if 'volume' in data.columns:
        data['volume_sma_20'] = data['volume'].astype(float).rolling(window=20, min_periods=1).mean()
    else:
        data['volume_sma_20'] = 1.0

    # 8. Bollinger Bands & BBW (20, 2.0)
    bb = calculate_bollinger_bands(close, period=20, num_std=2.0)
    data['bb_upper'] = bb.upper
    data['bb_middle'] = bb.middle
    data['bb_lower'] = bb.lower
    data['bbw'] = calculate_bbw(close, period=20, num_std=2.0)
    data['bbw_sma'] = data['bbw'].rolling(window=20, min_periods=1).mean()

    return data



def compute_indicators_from_candles(candles: Sequence[Any]) -> pd.DataFrame:
    """Convert a sequence of Candle objects or dicts into an enriched indicator DataFrame."""
    if not candles:
        return pd.DataFrame()

    records = []
    for c in candles:
        if isinstance(c, dict):
            records.append({
                'timestamp': c.get('timestamp'),
                'open': float(c.get('open', 0.0)),
                'high': float(c.get('high', 0.0)),
                'low': float(c.get('low', 0.0)),
                'close': float(c.get('close', 0.0)),
                'volume': float(c.get('volume', 0.0)),
            })
        else:
            records.append({
                'timestamp': getattr(c, 'timestamp', None),
                'open': float(getattr(c, 'open', 0.0)),
                'high': float(getattr(c, 'high', 0.0)),
                'low': float(getattr(c, 'low', 0.0)),
                'close': float(getattr(c, 'close', 0.0)),
                'volume': float(getattr(c, 'volume', 0.0)),
            })

    df = pd.DataFrame.from_records(records)
    return compute_all_indicators(df)
