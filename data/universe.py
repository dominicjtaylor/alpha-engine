"""
UK equity universe definitions for the Systematic Trading Research Framework.

All tickers use the Yahoo Finance .L suffix for London Stock Exchange securities.
The universe is a static list of FTSE 100 and FTSE 250 constituents.

Note: This is a fixed list and does not account for historical index membership
changes (survivorship bias). For production use, point-in-time constituent data
from a data vendor is recommended.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FTSE 100 constituents (approximate current list, .L suffix for Yahoo Finance)
# ---------------------------------------------------------------------------
FTSE100_TICKERS: list[str] = [
    "AAL.L", "ABF.L", "ADM.L", "AHT.L", "AIA.L", "ANTO.L", "AZN.L",
    "BA.L", "BARC.L", "BATS.L", "BDEV.L", "BEZ.L", "BKG.L", "BME.L",
    "BNZL.L", "BP.L", "BRBY.L", "BT-A.L", "CCH.L", "CNA.L", "CPG.L",
    "CRDA.L", "CRH.L", "DCC.L", "DGE.L", "DPLM.L", "EDV.L", "ENT.L",
    "EXPN.L", "EZJ.L", "FCIT.L", "FERG.L", "FLTR.L", "FRES.L", "GLEN.L",
    "GSK.L", "HLMA.L", "HLN.L", "HSBA.L", "HSX.L", "IAG.L", "ICG.L",
    "IHG.L", "III.L", "IMB.L", "INF.L", "ITRK.L", "JD.L", "KGF.L",
    "LAND.L", "LGEN.L", "LLOY.L", "LMP.L", "LSEG.L", "MKS.L", "MNDI.L",
    "MNG.L", "MRO.L", "NG.L", "NWG.L", "NXT.L", "OCDO.L", "PHNX.L",
    "PRU.L", "PSH.L", "PSN.L", "PSON.L", "RB.L", "RDSA.L", "REL.L",
    "RIO.L", "RKT.L", "RMV.L", "RR.L", "RSA.L", "RTO.L", "SBRY.L",
    "SDR.L", "SGE.L", "SGRO.L", "SHEL.L", "SKG.L", "SMDS.L", "SMIN.L",
    "SMT.L", "SN.L", "SPX.L", "SSE.L", "STAN.L", "STJ.L", "SVT.L",
    "TSCO.L", "TW.L", "ULVR.L", "UU.L", "VOD.L", "WEIR.L", "WPP.L",
    "WTB.L",
]

# ---------------------------------------------------------------------------
# FTSE 250 constituents (approximate current list, .L suffix)
# ---------------------------------------------------------------------------
FTSE250_TICKERS: list[str] = [
    "ABDN.L", "AGK.L", "AGR.L", "AJB.L", "APAX.L", "ATG.L", "ATL.L",
    "AUTO.L", "AV.L", "AWE.L", "BAB.L", "BBY.L", "BCPT.L", "BEMO.L",
    "BGB.L", "BGS.L", "BIFF.L", "BLND.L", "BOO.L", "BOWL.L", "BRW.L",
    "BSD.L", "BNKR.L", "BYG.L", "CAPC.L", "CAR.L", "CBG.L", "CBPE.L",
    "CCR.L", "CINE.L", "CLLN.L", "CLDN.L", "CMC.L", "CMCX.L", "COA.L",
    "COG.L", "CPI.L", "CSP.L", "CTY.L", "DARK.L", "DCG.L", "DELT.L",
    "DLN.L", "DMGT.L", "DNLM.L", "DOC.L", "DRX.L", "DSCV.L", "DUTY.L",
    "DVT.L", "ELECTRA.L", "ELM.L", "EMG.L", "EPWN.L", "ESL.L", "ESNT.L",
    "ESRT.L", "ETI.L", "EVET.L", "EWI.L", "FENR.L", "FGT.L", "FIF.L",
    "FLO.L", "FNX.L", "FORD.L", "FSTA.L", "FXI.L", "GAW.L", "GCP.L",
    "GDWN.L", "GEMD.L", "GLE.L", "GNK.L", "GPOR.L", "GRG.L", "GROW.L",
    "GRP.L", "HAT.L", "HBR.L", "HGT.L", "HIK.L", "HILTON.L", "HMSO.L",
    "HOC.L", "HSL.L", "HTG.L", "HWDN.L", "HYR.L", "IDH.L", "IMI.L",
    "INCH.L", "IPX.L", "IQE.L", "ITX.L", "IWG.L", "JEL.L", "JFJ.L",
    "JMG.L", "JMAT.L", "JTC.L", "KAZ.L", "KCIT.L", "KIE.L", "KWS.L",
    "LAD.L", "LBG.L", "LGO.L", "LMIN.L", "LRD.L", "MCRO.L", "MDC.L",
    "MICT.L", "MIL.L", "MIRL.L", "MNKS.L", "MOG.L", "MPI.L", "MPAC.L",
    "MTO.L", "MWE.L", "MWIG.L", "MXC.L", "NAV.L", "NCC.L", "NCYF.L",
    "NEX.L", "NFT.L", "NHC.L", "NMC.L", "NTOG.L", "NVT.L", "OCSL.L",
    "OML.L", "OPG.L", "OSB.L", "PCGE.L", "PETS.L", "PHI.L", "PLY.L",
    "PMGR.L", "PNN.L", "POG.L", "POLR.L", "POST.L", "PPB.L", "PPC.L",
    "PPHE.L", "PRTC.L", "PZC.L", "QQ.L", "QRT.L", "RAT.L", "RCP.L",
    "RDI.L", "RDSB.L", "REX.L", "RHIM.L", "RKH.L", "RKLB.L", "RNK.L",
    "ROC.L", "ROOF.L", "RPS.L", "RQI.L", "RSW.L", "SAE.L", "SAFE.L",
    "SAGA.L", "SCP.L", "SCT.L", "SDY.L", "SEA.L", "SFR.L", "SHED.L",
    "SHI.L", "SIG.L", "SIMR.L", "SKY.L", "SLA.L", "SLI.L", "SLPE.L",
    "SQZ.L", "SRC.L", "SRP.L", "SRET.L", "SRS.L", "SRX.L", "STHW.L",
    "STS.L", "SUPR.L", "SWJ.L", "SYNT.L", "TAL.L", "TCP.L", "TCAP.L",
    "TEM.L", "TLW.L", "TPFG.L", "TPS.L", "TRE.L", "TRIG.L", "TRN.L",
    "TYMN.L", "TYT.L", "UK5.L", "ULE.L", "ULVR.L", "UPR.L", "VCT.L",
    "VCP.L", "VEIL.L", "VLX.L", "VMUK.L", "VPC.L", "VRS.L", "VSL.L",
    "VSVS.L", "VTEC.L", "WAND.L", "WHR.L", "WIX.L", "WKF.L", "WLDN.L",
    "WPS.L", "XPP.L", "YCA.L", "YELL.L",
]

# Combined universe (deduplicated)
FTSE_ALL_TICKERS: list[str] = list(dict.fromkeys(FTSE100_TICKERS + FTSE250_TICKERS))


def get_ftse_universe(index: str = "ftse_all", top_n: int = 350) -> list[str]:
    """
    Return a list of LSE tickers with Yahoo Finance .L suffix.

    Parameters
    ----------
    index : str
        Universe to use. One of 'ftse100', 'ftse250', or 'ftse_all'.
    top_n : int
        Maximum number of tickers to return.

    Returns
    -------
    list[str]
        Ticker symbols with .L suffix, e.g. ["AZN.L", "HSBA.L", ...].
    """
    if index == "ftse100":
        tickers = FTSE100_TICKERS
    elif index == "ftse250":
        tickers = FTSE250_TICKERS
    elif index == "ftse_all":
        tickers = FTSE_ALL_TICKERS
    else:
        raise ValueError(f"Unknown universe '{index}'. Choose 'ftse100', 'ftse250', or 'ftse_all'.")

    result = tickers[:top_n]
    logger.info("Universe '%s' loaded: %d tickers (capped at %d)", index, len(result), top_n)
    return result


def load_ftse_universe(index: str = "ftse_all", top_n: int = 350) -> list[str]:
    """
    Primary entry point for loading the UK equity universe.

    Ensures all returned tickers use the .L suffix required by Yahoo Finance
    for London Stock Exchange securities.

    Parameters
    ----------
    index : str
        'ftse100', 'ftse250', or 'ftse_all'.
    top_n : int
        Maximum universe size.

    Returns
    -------
    list[str]
        Ticker list with .L suffix.
    """
    tickers = get_ftse_universe(index=index, top_n=top_n)
    # Validate all tickers have .L suffix
    invalid = [t for t in tickers if not t.endswith(".L")]
    if invalid:
        logger.warning("%d tickers missing .L suffix: %s", len(invalid), invalid[:5])
    return tickers


def load_universe_from_file(path: str) -> list[str]:
    """
    Load ticker symbols from a newline-delimited text file.

    Each line should contain one ticker symbol. Lines starting with '#'
    are treated as comments and ignored. Tickers without .L suffix have
    it appended automatically.

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    list[str]
        List of ticker symbols with .L suffix.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    tickers = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.endswith(".L"):
                line = f"{line}.L"
            tickers.append(line)

    logger.info("Loaded %d tickers from %s", len(tickers), path)
    return tickers


def apply_liquidity_filters(
    tickers: list[str],
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    min_price_gbp: float = 0.50,
    min_avg_volume_gbp: float = 1_000_000.0,
    lookback_days: int = 63,
) -> list[str]:
    """
    Filter out illiquid and low-priced securities from the universe.

    Parameters
    ----------
    tickers : list[str]
        Candidate tickers to filter.
    prices : pd.DataFrame
        Adjusted close prices, wide format (index=date, columns=tickers).
    volumes : pd.DataFrame, optional
        Daily trading volumes. If provided, used for volume filter.
        If None, only price filter is applied.
    min_price_gbp : float
        Minimum average price in GBP over the lookback period.
    min_avg_volume_gbp : float
        Minimum average daily volume in GBP (price × volume proxy).
    lookback_days : int
        Number of recent trading days to compute averages over.

    Returns
    -------
    list[str]
        Filtered list of tickers passing all liquidity criteria.
    """
    available = [t for t in tickers if t in prices.columns]
    if not available:
        logger.warning("No tickers found in price data for liquidity filtering.")
        return tickers

    recent_prices = prices[available].tail(lookback_days)
    avg_prices = recent_prices.mean()

    # Price filter
    price_pass = avg_prices[avg_prices >= min_price_gbp].index.tolist()
    removed_by_price = len(available) - len(price_pass)
    if removed_by_price > 0:
        logger.info("Liquidity filter: removed %d tickers below £%.2f avg price",
                    removed_by_price, min_price_gbp)

    # Volume filter (proxy: use volume DataFrame if provided)
    if volumes is not None:
        vol_available = [t for t in price_pass if t in volumes.columns]
        recent_volumes = volumes[vol_available].tail(lookback_days)
        recent_prices_filtered = prices[vol_available].tail(lookback_days)
        # Approximate GBP volume = price × share volume
        avg_gbp_volume = (recent_prices_filtered * recent_volumes).mean()
        vol_pass = avg_gbp_volume[avg_gbp_volume >= min_avg_volume_gbp].index.tolist()
        removed_by_vol = len(price_pass) - len(vol_pass)
        if removed_by_vol > 0:
            logger.info("Liquidity filter: removed %d tickers below £%.0f avg daily volume",
                        removed_by_vol, min_avg_volume_gbp)
        result = vol_pass
    else:
        result = price_pass

    logger.info("Liquidity filter: %d/%d tickers passed", len(result), len(available))
    return result
