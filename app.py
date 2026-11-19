"""ValorIA: guided 5-step property valuation for Madrid (single-file version).

Run:  streamlit run app.py   (the previous single-page app is kept as app_classic.py)
Needs the rest of the repo next to it (src/, data/, models/); it replaces the ui/ folder.
"""
from __future__ import annotations

import html
import io
import re
from datetime import date
import base64
import hashlib
import json
import time
from datetime import datetime
import numpy as np
from src.config import LLM_MODEL, MARKET_ADJUSTMENT, MODEL_PATH, ROOT, USE_LLM
from src.extract import BOOL_FIELDS, extract_features
from src.features import _haversine_km, distances
from src.pipeline import CONDITION_FLAGS
from src.report import eur
from src.valuation import fill_missing, find_comparables, value_flat
from src.vision import assess_photos
import streamlit as st
from html import escape
import pandas as pd


# ============================================================================
# src/pdf_report.py: Branded one-page PDF of the valuation report (uses reportlab).
# ============================================================================
P_NAVY, P_BURGUNDY, P_MUTED, P_LINE = ('#14213D', '#7D2E3A', '#6B6B6B', '#E3DED6')

def P__inline(text: str) -> str:
    text = html.escape(text.replace('−', '-').replace('–', '-'), quote=False)
    return re.sub('\\*\\*(.+?)\\*\\*', '<b>\\1</b>', text)

def P__fonts() -> tuple[str, str]:
    """Vera ships with reportlab and, unlike Helvetica, has the ² glyph."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        if 'Vera' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('Vera', 'Vera.ttf'))
            pdfmetrics.registerFont(TTFont('VeraBd', 'VeraBd.ttf'))
        return ('Vera', 'VeraBd')
    except Exception:
        return ('Helvetica', 'Helvetica-Bold')

def P_build_pdf(ctx: dict, report_md: str, eur) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None
    navy, burgundy, muted, line = (colors.HexColor(c) for c in (P_NAVY, P_BURGUNDY, P_MUTED, P_LINE))
    regular, bold = P__fonts()
    base = ParagraphStyle('base', fontName=regular, fontSize=9.5, leading=13.5, textColor=navy)
    styles = {'h1': ParagraphStyle('h1', parent=base, fontName=bold, fontSize=20, leading=24), 'h2': ParagraphStyle('h2', parent=base, fontName=bold, fontSize=11.5, textColor=burgundy, spaceBefore=8, spaceAfter=2), 'small': ParagraphStyle('small', parent=base, fontSize=7.5, leading=10, textColor=muted), 'big': ParagraphStyle('big', parent=base, fontName=bold, fontSize=24, leading=28), 'label': ParagraphStyle('label', parent=base, fontSize=8, textColor=muted)}
    f, v, vd = (ctx['features'], ctx['valuation'], ctx.get('verdict'))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(line)
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.setFont(regular, 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 11 * mm, 'ValorIA · Automated pre-valuation, not an official appraisal (tasación).')
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f'Page {doc.page}')
        canvas.restoreState()
    story = [Paragraph(f"Valor<font color='{P_BURGUNDY}'>IA</font>", styles['h1']), Paragraph(f"Property valuation report · {date.today():%d %B %Y} · {html.escape(ctx['location']['label'])}", styles['small']), Spacer(1, 6 * mm)]
    cells = [[Paragraph('ESTIMATED MARKET VALUE', styles['label']), Paragraph('PRICE PER M²', styles['label']), Paragraph('ASKING PRICE', styles['label'])], [Paragraph(eur(v['estimate']), styles['big']), Paragraph(eur(v['per_m2']), styles['h2']), Paragraph(f"{eur(f['asking_price'])} ({vd['pct']:+.0%})" if vd else 'Not provided', styles['h2'])], [Paragraph(f"Model tree spread {eur(v['low'])} - {eur(v['high'])}", styles['small']), '', Paragraph(html.escape(vd['label']) if vd else '', styles['small'])]]
    t = Table(cells, colWidths=[78 * mm, 45 * mm, 51 * mm])
    t.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.6, line), ('BACKGROUND', (0, 0), (-1, -1), colors.white), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, 0), 7), ('BOTTOMPADDING', (0, -1), (-1, -1), 7)]))
    story += [t, Spacer(1, 4 * mm)]
    bullets = []
    for raw in report_md.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(('- ', '* ')):
            bullets.append(ListItem(Paragraph(P__inline(s[2:]), base), leftIndent=10))
            continue
        if bullets:
            story.append(ListFlowable(bullets, bulletType='bullet', start='•', leftIndent=10))
            bullets = []
        if s.startswith('#'):
            story.append(Paragraph(P__inline(s.lstrip('#').strip()), styles['h2']))
        else:
            story.append(Paragraph(P__inline(s), base))
    if bullets:
        story.append(ListFlowable(bullets, bulletType='bullet', start='•', leftIndent=10))
    if ctx.get('comps'):
        story += [Paragraph('Comparable flats (idealista18 dataset)', styles['h2'])]
        rows = [['Area', 'Size', 'Bed', 'Bath', 'Price', '€/m²', 'Distance']]
        rows += [[c['near'], f"{c['area_m2']:.0f} m²", c['rooms'], c['baths'], eur(c['price']), eur(c['price_per_m2']), f"{c['distance_km']:.2f} km"] for c in ctx['comps']]
        ct = Table(rows, colWidths=[36 * mm, 20 * mm, 12 * mm, 12 * mm, 32 * mm, 28 * mm, 22 * mm])
        ct.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), regular, 8.5), ('FONT', (0, 0), (-1, 0), bold, 8.5), ('TEXTCOLOR', (0, 0), (-1, -1), navy), ('LINEBELOW', (0, 0), (-1, -1), 0.4, line), ('ALIGN', (2, 0), (-1, -1), 'RIGHT')]))
        story.append(ct)
    story += [Spacer(1, 5 * mm), Paragraph('Method: features read from the listing (and photos, if provided) by a language model; price computed by a random forest trained on 2018 Madrid listings (idealista18, ODbL), multiplied by a market adjustment; comparables found by nearest-neighbour search. Prices of comparables are 2018 asking prices × the same adjustment.', styles['small'])]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=22 * mm, title='ValorIA valuation report').build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ============================================================================
# src/engine.py: Engine for the guided app: runs the existing pipeline steps one at a time.
# ============================================================================
E_DEMO_DIR = ROOT / 'data' / 'demo_listings'
E_SNAPSHOT_DIR = ROOT / 'data' / 'demo_runs'
E_METRICS_PATH = MODEL_PATH.parent / 'metrics.json'
E_NEIGHBOURHOODS = {'Sol (centre)': (40.4168, -3.7038), 'Chamberí': (40.434, -3.704), 'Salamanca': (40.43, -3.678), 'Retiro': (40.411, -3.676), 'Chamartín': (40.458, -3.677), 'Malasaña': (40.426, -3.705), 'Lavapiés': (40.4085, -3.701), 'Argüelles': (40.43, -3.718), 'Tetuán': (40.46, -3.698), 'Arganzuela': (40.398, -3.699), 'Carabanchel': (40.383, -3.728), 'Puente de Vallecas': (40.391, -3.658)}
E_EXAMPLES = {'chamberi': {'title': 'Chamberí family flat', 'file': 'example_listing.txt', 'hood': 'Chamberí', 'asking': 520000, 'blurb': '85 m², 3 bedrooms, renovated, lift and doorman'}, 'lavapies': {'title': 'Lavapiés interior flat, priced high', 'file': 'overpriced_listing.txt', 'hood': 'Lavapiés', 'asking': 365000, 'blurb': '55 m², interior, no lift: shows how overpricing is flagged'}}
E_STAGES = [('extract', 'Listing description', 'LLM feature extraction', 'GenAI'), ('vision', 'Property photos', 'Vision model analysis', 'GenAI'), ('predict', 'Structured features', 'ML price prediction', 'Machine learning'), ('comps', 'Madrid property database', 'Comparable retrieval', 'Retrieval'), ('report', 'Valuation results', 'LLM report generation', 'GenAI')]
E_NUMERIC_FIELDS = {'area_m2': ('Size', '📐'), 'rooms': ('Bedrooms', '🛏️'), 'baths': ('Bathrooms', '🛁'), 'floor': ('Floor', '🏢'), 'year_built': ('Year built', '🗓️')}
E_AMENITIES = {'has_lift': 'Lift', 'is_exterior': 'Exterior', 'has_terrace': 'Terrace / balcony', 'has_air_conditioning': 'Air conditioning', 'has_parking': 'Parking', 'has_storage_room': 'Storage room', 'has_built_in_wardrobe': 'Built-in wardrobes', 'has_doorman': 'Doorman', 'has_pool': 'Pool', 'has_garden': 'Garden', 'is_top_floor': 'Top floor / penthouse', 'is_duplex': 'Duplex', 'is_studio': 'Studio', 'orientation_south': 'South facing', 'orientation_north': 'North facing', 'orientation_east': 'East facing', 'orientation_west': 'West facing'}
E_CONDITIONS = {'new': 'New / like new', 'good': 'Good, lived-in', 'to_renovate': 'Needs renovation', 'unknown': 'Unknown'}
E_MODEL_DEFAULTS = {'floor': 1, 'year_built': 1975, 'baths': 1}
E_SOURCE_LABELS = {'user': 'Provided by you', 'listing': 'From listing text', 'photo': 'From photos', 'edited': 'Corrected by you', 'missing': 'Missing'}

def E_mode_label() -> str:
    return f'AI connected · {LLM_MODEL}' if USE_LLM else 'Offline mode · keyword rules'

def E_fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]

# Example texts are built into this file, so no extra files are needed
E_EXAMPLE_TEXTS = {'example_listing.txt': 'Piso exterior en venta en Chamberí, 85 m2, 3 habitaciones y 2 baños.\nCuarta planta con ascensor. Reformado en 2021, cocina equipada,\naire acondicionado y armarios empotrados. Orientación sur, muy luminoso.\nFinca de 1960 con portero. Trastero incluido.\nPrecio: 520.000 €',
                   'overpriced_listing.txt': 'Piso interior en venta en Lavapiés, 55 m2, 2 habitaciones y 1 baño.\nPlanta baja en finca antigua de 1920, en buen estado.\nCocina independiente y suelos de madera. A cinco minutos del metro.\nPrecio: 365.000 €'}


def E_load_example(key: str) -> dict:
    ex = E_EXAMPLES[key]
    return {**ex, 'text': E_EXAMPLE_TEXTS[ex['file']]}


def E_saved_listings() -> dict:
    """Real listings saved by the team in data/demo_listings/<name>/ (text, info.json, photos)."""
    try:
        from src.demo_listings import list_demos
        return list_demos()
    except Exception:
        return {}

def E_load_saved_listing(folder) -> dict:
    from src.demo_listings import load_demo
    d = load_demo(folder)
    hood = d['neighbourhood'] if d['neighbourhood'] in E_NEIGHBOURHOODS else None
    return {'text': d['text'], 'hood': hood, 'address': d['address'], 'asking': int(d['asking_price']), 'coords': d['coords'], 'photos': d['photos'], 'photo_names': d['photo_names'], 'neighbourhood': d['neighbourhood']}

def E_friendly_error(exc: Exception) -> str:
    """Turn an exception into a short, non-technical message (never shows keys)."""
    name, msg = (type(exc).__name__, str(exc).lower())
    if isinstance(exc, FileNotFoundError) and 'model' in msg:
        return 'The price model has not been trained yet. Run `python -m src.train_model` once, then retry.'
    if 'authentication' in name.lower() or '401' in msg or 'api key' in msg or ('api_key' in msg):
        return 'The AI service rejected the API key. Check the key in your .env file and restart the app.'
    if 'ratelimit' in name.lower() or '429' in msg or 'quota' in msg:
        return 'The AI service is rate-limited or out of credit. Wait a moment, check your usage limits, then retry.'
    if 'timeout' in name.lower() or 'timed out' in msg or 'connection' in name.lower() or ('connect' in msg):
        return 'Could not reach the AI service (network problem or timeout). Check the internet connection and retry.'
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return 'The AI returned an answer that could not be read. Retrying usually fixes this.'
    return f'Something went wrong ({name}). Your inputs are kept, so you can retry.'

def E__record(status: str, detail: str='', seconds: float | None=None) -> dict:
    return {'status': status, 'detail': detail, 'seconds': seconds}

def E_empty_stages() -> dict:
    return {key: E__record('pending') for key, *_ in E_STAGES}

def E_analyse_listing(text: str, use_keywords: bool=False) -> tuple[dict, dict]:
    """Listing text -> features. use_keywords=True forces the offline keyword parser."""
    t0 = time.perf_counter()
    if use_keywords or not USE_LLM:
        from src.extract import _clean, _keyword_extract
        feats = _clean(_keyword_extract(text))
        how = 'Keyword rules (offline fallback, no LLM)' if USE_LLM else 'Keyword rules (offline mode, no LLM)'
    else:
        feats = extract_features(text)
        how = f'LLM extraction with {LLM_MODEL}'
    return (feats, E__record('completed', how, time.perf_counter() - t0))

def E_analyse_photos(images: list[bytes]) -> tuple[dict | None, dict]:
    if not images:
        return (None, E__record('skipped', 'No photos uploaded'))
    if not USE_LLM:
        return (None, E__record('skipped', 'Photo analysis needs an API key (offline mode)'))
    t0 = time.perf_counter()
    photo = assess_photos(images)
    n = min(len(images), 4)
    return (photo, E__record('completed', f'{LLM_MODEL} looked at {n} photo(s)', time.perf_counter() - t0))

def E_merge_inputs(extracted: dict, overrides: dict, photo: dict | None, asking: float | None):
    """Combine extracted features, user-typed values and photo results.
    Returns (features, provenance) where provenance[field] is a SOURCE_LABELS key."""
    feats = dict(extracted)
    prov = {}
    for key in E_NUMERIC_FIELDS:
        prov[key] = 'listing' if feats.get(key) not in (None, '') else 'missing'
    for key, val in (overrides or {}).items():
        if val not in (None, 0, ''):
            feats[key], prov[key] = (val, 'user')
    for key in E_AMENITIES:
        feats[key] = int(feats.get(key) or 0)
        prov[key] = 'listing' if feats[key] else 'missing'
    cond = feats.get('condition') or 'unknown'
    prov['condition'] = 'listing' if cond != 'unknown' else 'missing'
    if cond == 'unknown' and photo and (photo.get('condition') in E_CONDITIONS):
        feats['condition'], prov['condition'] = (photo['condition'], 'photo')
    if asking:
        feats['asking_price'], prov['asking_price'] = (float(asking), 'user')
    elif feats.get('asking_price'):
        prov['asking_price'] = 'listing'
    else:
        prov['asking_price'] = 'missing'
    return (feats, prov)

def E_prepare(features: dict, lat: float, lon: float) -> dict:
    f = dict(features)
    f['condition_new'], f['condition_to_renovate'] = CONDITION_FLAGS.get(f.get('condition'), (0, 0))
    f.update(latitude=lat, longitude=lon, **distances(lat, lon))
    return fill_missing(f)

def E_predict(features: dict) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    v = value_flat(features)
    return (v, E__record('completed', 'Random forest (scikit-learn) trained on idealista18 data', time.perf_counter() - t0))

def E_sensitivity(features: dict, base: float) -> list[dict]:
    """Measured by the model itself: how the estimate changes when ONE input is
    switched, everything else equal. Random forests capture interactions, so
    these effects do not add up exactly to the total."""
    tests = []
    for key, label in E_AMENITIES.items():
        if features.get(key):
            tests.append((f'{label}', {key: 0}, 'present'))
    if (features.get('baths') or 0) > 1:
        tests.append((f"{int(features['baths'])} bathrooms", {'baths': features['baths'] - 1}, 'one fewer'))
    if features.get('area_m2'):
        tests.append((f"Size ({features['area_m2']:.0f} m²)", {'area_m2': features['area_m2'] - 10}, '10 m² smaller'))
    cond = features.get('condition')
    if cond in ('new', 'to_renovate'):
        tests.append((E_CONDITIONS[cond], {'condition_new': 0, 'condition_to_renovate': 0}, 'vs. good condition'))
    out = []
    for label, change, compared in tests:
        alt = value_flat({**features, **change})['estimate']
        delta = base - alt
        if abs(delta) >= max(500.0, 0.01 * base):  # under 1% of the price = no meaningful effect
            out.append({'label': label, 'delta': float(delta), 'compared': compared})
    return sorted(out, key=lambda d: -abs(d['delta']))

def E_verdict(asking: float | None, v: dict, tolerance: float=0.05) -> dict | None:
    if not asking:
        return None
    diff = asking / v['estimate'] - 1
    if diff > tolerance:
        label, tone = ('Above estimated market value', 'warn')
    elif diff < -tolerance:
        label, tone = ('Below estimated market value', 'good')
    else:
        label, tone = ('Approximately aligned with estimated market value', 'neutral')
    return {'label': label, 'tone': tone, 'pct': diff, 'eur': asking - v['estimate'], 'outside_range': asking > v['high'] or asking < v['low'], 'tolerance': tolerance}

def E_comparables(features: dict) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    comps = find_comparables(features)
    lat, lon = (features['latitude'], features['longitude'])
    names, centres = (list(E_NEIGHBOURHOODS), np.array(list(E_NEIGHBOURHOODS.values())))
    rows = []
    for _, c in comps.iterrows():
        dist = float(_haversine_km(lat, lon, [c['latitude']], [c['longitude']])[0])
        near = _haversine_km(c['latitude'], c['longitude'], centres[:, 0], centres[:, 1])
        rows.append({'area_m2': float(c['area_m2']), 'rooms': int(c['rooms']), 'baths': int(c['baths']), 'has_lift': int(c['has_lift']), 'has_terrace': int(c['has_terrace']), 'price': float(c['price']), 'price_per_m2': float(c['price_per_m2']), 'latitude': float(c['latitude']), 'longitude': float(c['longitude']), 'distance_km': dist, 'near': names[int(near.argmin())] if near.min() < 1.5 else 'Madrid', 'size_diff': float(c['area_m2'] - (features.get('area_m2') or 0))})
    rec = E__record('completed', f'{len(rows)} nearest flats by location, size and rooms (nearest-neighbour search over idealista18)', time.perf_counter() - t0)
    return (rows, rec)
E_REPORT_PROMPT = 'You are ValorIA, an assistant for real estate agents in Madrid.\nWrite a professional one-page valuation report in Markdown (max 380 words) with\nexactly these sections as ## headings:\n1. Property overview\n2. Estimated market value\n3. Key characteristics\n4. Main valuation considerations\n5. Comparable properties\n6. Asking price comparison\n7. Limitations\nRules: use ONLY the facts given; never change, round differently or invent numbers.\nIf a detail is marked as estimated, write that it was estimated from similar flats\n(e.g. "built around 1950, estimated"); never present it as a fact from the listing.\nThe price comes from a machine learning model, not from you. If a section has no\ndata (for example no asking price), say so in one sentence. Do not give investment\nadvice. Write in {language}.'

def E_facts_for_report(ctx: dict) -> str:
    f, v = (ctx['features'], ctx['valuation'])
    lines = [f"Location: {ctx['location']['label']} (from {ctx['location']['source']})", f"Size {f.get('area_m2')} m², {f.get('rooms')} bedrooms, {f.get('baths')} bathrooms, floor {f.get('floor')}, year built {f.get('year_built')}, condition: {E_CONDITIONS.get(f.get('condition'), 'unknown')}", 'Features mentioned: ' + (', '.join((E_AMENITIES[k] for k in E_AMENITIES if f.get(k))) or 'none'), f"ML model estimate: {eur(v['estimate'])} ({eur(v['per_m2'])}/m²)", f"Spread of the random forest's trees (10th-90th percentile): {eur(v['low'])} to {eur(v['high'])}", f'Prices are 2018 listing prices multiplied by a market adjustment of x{MARKET_ADJUSTMENT}', f"Distance to city centre {f.get('dist_center_km', 0):.1f} km, nearest metro {f.get('dist_metro_km', 0):.2f} km"]
    if ctx.get('drivers'):
        lines.append('Model sensitivity (value each item adds or removes, all else equal): ' + '; '.join((f"{d['label']} {('+' if d['delta'] >= 0 else '-')}{eur(abs(d['delta']))}" for d in ctx['drivers'][:6])))
    if ctx.get('comps'):
        med = float(np.median([c['price_per_m2'] for c in ctx['comps']]))
        lines.append(f"{len(ctx['comps'])} comparable flats from the 2018 dataset: median {eur(med)}/m²; " + '; '.join((f"{c['area_m2']:.0f} m², {c['rooms']} bed, {eur(c['price'])}, {c['distance_km']:.1f} km away" for c in ctx['comps'])))
    if f.get('estimated'):
        lines.append('NOT in the listing, estimated from the 15 most similar flats in our dataset (say so, never state as fact): ' + ', '.join((f"{k.replace('_', ' ')} = {f.get(k)}" for k in f['estimated'])))
    if f.get('highlights'):
        lines.append('Listing highlights: ' + '; '.join(f['highlights']))
    if ctx.get('photo') and ctx['photo'].get('notes'):
        lines.append(f"Photo assessment: {ctx['photo']['notes']}")
    vd = ctx.get('verdict')
    if vd:
        lines.append(f"Asking price {eur(f['asking_price'])}: {vd['pct']:+.0%} vs estimate ({vd['label']}; tolerance ±{vd['tolerance']:.0%})")
    else:
        lines.append('No asking price given.')
    return '\n'.join((f'- {line}' for line in lines))

def E__num(x) -> str:
    return '?' if x in (None, '') else f'{float(x):.0f}'

def E_template_report(ctx: dict) -> str:
    """Report built from the same facts without an LLM (offline mode or LLM failure)."""
    f, v, vd = (ctx['features'], ctx['valuation'], ctx.get('verdict'))
    amen = ', '.join((E_AMENITIES[k] for k in E_AMENITIES if f.get(k))) or 'none mentioned'
    out = ['## Property overview', f"{E__num(f.get('area_m2'))} m² flat in {ctx['location']['label']}, {E__num(f.get('rooms'))} bedrooms and {E__num(f.get('baths'))} bathrooms. Condition: {E_CONDITIONS.get(f.get('condition'), 'unknown').lower()}." + (' Estimated from similar flats in our dataset (not in the listing): ' + ', '.join((f"{k.replace('_', ' ')} {E__num(f.get(k))}" for k in f['estimated'])) + '.' if f.get('estimated') else ''), '## Estimated market value', f"**{eur(v['estimate'])}** ({eur(v['per_m2'])}/m²). The model's trees range from {eur(v['low'])} to {eur(v['high'])} (10th–90th percentile).", '## Key characteristics', f'Features: {amen}.', '## Main valuation considerations']
    if ctx.get('drivers'):
        out += [f"- {d['label']}: {('+' if d['delta'] >= 0 else '−')}{eur(abs(d['delta']))} ({d['compared']}, model sensitivity)" for d in ctx['drivers'][:5]]
    else:
        out.append('No single characteristic changed the estimate noticeably.')
    out.append('## Comparable properties')
    if ctx.get('comps'):
        med = float(np.median([c['price_per_m2'] for c in ctx['comps']]))
        out.append(f"{len(ctx['comps'])} similar flats nearby in the 2018 dataset have a median of {eur(med)}/m², against {eur(v['per_m2'])}/m² for this property.")
    else:
        out.append('No comparable flats were retrieved.')
    out.append('## Asking price comparison')
    out.append(f"Asking price {eur(f['asking_price'])}, {vd['pct']:+.0%} versus the estimate: {vd['label'].lower()}." if vd else 'No asking price was provided.')
    out += ['## Limitations', E_LIMITATIONS]
    return '\n\n'.join(out)
E_LIMITATIONS = 'This is an automated pre-valuation, not an official appraisal (tasación). The model is trained on 2018 Madrid listing prices (idealista18), adjusted by a market factor, and cannot see everything a visit would reveal.'

def E_write_client_report(ctx: dict, language: str='English') -> tuple[str, dict]:
    if not USE_LLM:
        return (E_template_report(ctx), E__record('skipped', 'Offline mode: report built from a fixed template (no LLM)'))
    t0 = time.perf_counter()
    from src.llm import ask_text
    md = ask_text(E_REPORT_PROMPT.format(language=language), E_facts_for_report(ctx))
    return (md, E__record('completed', f"Written by {LLM_MODEL} from the pipeline's numbers", time.perf_counter() - t0))

def E_load_metrics() -> dict | None:
    try:
        return json.loads(E_METRICS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None

def E_model_size() -> dict:
    try:
        from src.valuation import _load
        b = _load()
        return {'n_rows': len(b['train_data']), 'n_trees': len(b['model'].estimators_)}
    except Exception:
        return {}

def E__thumb(img: bytes, size: int=640) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(img)).convert('RGB')
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=80)
    return buf.getvalue()

def E_save_snapshot(name: str, state: dict) -> str:
    E_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    data = {**state, 'saved_at': datetime.now().isoformat(timespec='minutes'), 'mode': E_mode_label(), 'market_adjustment': MARKET_ADJUSTMENT, 'photos': [base64.b64encode(E__thumb(p)).decode() for p in state.get('photos', [])[:4]]}
    slug = ''.join((ch if ch.isalnum() else '_' for ch in name.lower())).strip('_') or 'run'
    path = E_SNAPSHOT_DIR / f'{slug}.json'
    path.write_text(json.dumps(data, indent=1, default=str), encoding='utf-8')
    return path.name

def E_list_snapshots() -> list[str]:
    return sorted((p.stem for p in E_SNAPSHOT_DIR.glob('*.json'))) if E_SNAPSHOT_DIR.exists() else []

def E_load_snapshot(stem: str) -> dict:
    data = json.loads((E_SNAPSHOT_DIR / f'{stem}.json').read_text(encoding='utf-8'))
    data['photos'] = [base64.b64decode(p) for p in data.get('photos', [])]
    return data
E_MADRID_BOX = (40.3, 40.56, -3.89, -3.52)

def E_locate(address: str, hood: str) -> dict:
    """Geocode the address (OpenStreetMap Nominatim). Falls back to the neighbourhood centre."""
    warning = None
    if address and address.strip():
        query = address if 'madrid' in address.lower() else f'{address}, Madrid'
        try:
            from geopy.geocoders import Nominatim
            loc = Nominatim(user_agent='valoria-student-project').geocode(query, timeout=8)
            if loc and E_MADRID_BOX[0] <= loc.latitude <= E_MADRID_BOX[1] and (E_MADRID_BOX[2] <= loc.longitude <= E_MADRID_BOX[3]):
                return {'lat': loc.latitude, 'lon': loc.longitude, 'label': address.strip(), 'source': 'address (OpenStreetMap)', 'warning': None}
            warning = 'Address not found inside Madrid, so the neighbourhood centre is used instead.'
        except Exception:
            warning = 'Address lookup is unavailable (no internet?), so the neighbourhood centre is used instead.'
    lat, lon = E_NEIGHBOURHOODS[hood]
    return {'lat': lat, 'lon': lon, 'label': hood, 'source': 'neighbourhood centre (approximate)', 'warning': warning}


# ============================================================================
# ui/styles.py: Visual system for ValorIA: one CSS block injected once per run.
# ============================================================================
STY_CSS = '\n<style>\n@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap\');\n\n:root {\n  --bg:#F7F5F1; --card:#FFFFFF; --ink:#14213D; --muted:#6B6F76; --line:#E6E1D8;\n  --accent:#7D2E3A; --accent-soft:#F6ECED; --navy-soft:#EAEFF6;\n  --good:#2F6B4F; --good-soft:#E7F2EC; --warn:#9A5B13; --warn-soft:#FBF1E2;\n  --violet:#5B4B8A; --violet-soft:#EFEBF7; --grey-soft:#F0EEEA;\n}\nhtml, body, [class*="css"], .stMarkdown, button, input, textarea, select {\n  font-family: \'Inter\', -apple-system, BlinkMacSystemFont, \'Segoe UI\', sans-serif !important;\n}\n/* Hide default Streamlit chrome */\n[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,\n[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],\n[data-testid="collapsedControl"] { display:none !important; }\n.block-container { max-width: 1120px; padding-top: 1.1rem; padding-bottom: 2rem; }\nh1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }\n\n/* Top navigation */\n.v-nav { display:flex; align-items:center; gap:.9rem; padding:.2rem 0 .9rem; }\n.v-logo { width:38px; height:38px; border-radius:10px; background:var(--ink); color:#fff;\n  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.05rem;\n  box-shadow: inset 0 -3px 0 var(--accent); }\n.v-brand { font-weight:700; font-size:1.25rem; color:var(--ink); line-height:1.1; }\n.v-brand span { color:var(--accent); }\n.v-tagline { color:var(--muted); font-size:.82rem; }\n.v-pill { display:inline-flex; align-items:center; gap:.4rem; font-size:.76rem; font-weight:500;\n  padding:.28rem .7rem; border-radius:999px; border:1px solid var(--line); background:var(--card);\n  color:var(--ink); white-space:nowrap; }\n.v-dot { width:7px; height:7px; border-radius:50%; background:var(--good); }\n.v-dot.off { background:#B8B2A7; } .v-dot.demo { background:var(--warn); }\n\n/* Stepper */\n.v-stepper { display:flex; gap:.4rem; margin:.3rem 0 1.6rem; padding:.85rem 1rem; background:var(--card);\n  border:1px solid var(--line); border-radius:14px; }\n.v-step { flex:1; display:flex; align-items:center; gap:.55rem; min-width:0; }\n.v-step .n { flex:none; width:26px; height:26px; border-radius:50%; display:flex; align-items:center;\n  justify-content:center; font-size:.78rem; font-weight:600; border:1.5px solid var(--line);\n  color:var(--muted); background:var(--card); transition:all .25s ease; }\n.v-step .t { font-size:.8rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n.v-step .bar { flex:1; height:2px; background:var(--line); border-radius:2px; margin-left:.2rem; }\n.v-step.done .n { background:var(--ink); border-color:var(--ink); color:#fff; }\n.v-step.done .bar { background:var(--ink); }\n.v-step.active .n { border-color:var(--accent); color:var(--accent); box-shadow:0 0 0 4px var(--accent-soft); }\n.v-step.active .t { color:var(--ink); font-weight:600; }\n\n/* Page headers */\n.v-eyebrow { font-size:.74rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); }\n.v-title { font-family:\'Fraunces\', Georgia, serif; font-size:2.05rem; font-weight:600; color:var(--ink);\n  margin:.15rem 0 .3rem; letter-spacing:-.02em; line-height:1.15; }\n.v-sub { color:var(--muted); font-size:1rem; max-width:720px; margin-bottom:1.3rem; }\n\n/* Cards */\n.v-card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem;\n  height:100%; animation: v-fade .35s ease both; }\n.v-card.tight { padding:.8rem .95rem; }\n.v-card h4 { margin:0 0 .35rem; font-size:.95rem; color:var(--ink); }\n.v-card p { margin:0; color:var(--muted); font-size:.86rem; }\n@keyframes v-fade { from {opacity:0; transform:translateY(4px);} to {opacity:1; transform:none;} }\n.v-feat-label { font-size:.76rem; color:var(--muted); display:flex; align-items:center; gap:.35rem; }\n.v-feat-value { font-size:1.35rem; font-weight:600; color:var(--ink); margin:.15rem 0 .45rem; }\n.v-feat-value.missing { color:#B3ADA3; font-weight:500; font-size:1.05rem; }\n\n/* Badges / chips */\n.v-badge { display:inline-block; font-size:.68rem; font-weight:600; padding:.15rem .5rem; border-radius:6px; }\n.b-user { background:var(--navy-soft); color:var(--ink); }\n.b-listing { background:var(--good-soft); color:var(--good); }\n.b-photo { background:var(--violet-soft); color:var(--violet); }\n.b-edited { background:var(--warn-soft); color:var(--warn); }\n.b-missing { background:var(--grey-soft); color:var(--muted); }\n.b-genai { background:var(--violet-soft); color:var(--violet); }\n.b-ml { background:var(--navy-soft); color:var(--ink); }\n.v-chip { display:inline-flex; align-items:center; gap:.35rem; padding:.35rem .7rem; margin:0 .4rem .45rem 0;\n  border-radius:999px; font-size:.82rem; border:1px solid var(--line); background:var(--card); color:var(--ink); }\n.v-chip.on { border-color:#CFE3D7; background:var(--good-soft); }\n\n/* Valuation hero */\n.v-hero { background:var(--ink); color:#fff; border-radius:20px; padding:1.7rem 1.9rem; position:relative;\n  overflow:hidden; animation: v-fade .4s ease both; }\n.v-hero:after { content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;\n  border-radius:50%; background:radial-gradient(circle, rgba(125,46,58,.55), transparent 70%); }\n.v-hero .lbl { font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#C9CFDB; }\n.v-hero .big { font-family:\'Fraunces\', Georgia, serif; font-size:3.3rem; font-weight:600; line-height:1.05; margin:.25rem 0 .4rem; }\n.v-hero .meta { color:#C9CFDB; font-size:.86rem; max-width:560px; }\n.v-kpi .k { font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }\n.v-kpi .v { font-size:1.35rem; font-weight:600; color:var(--ink); margin-top:.2rem; }\n.v-kpi .s { font-size:.78rem; color:var(--muted); margin-top:.15rem; }\n\n/* Verdict */\n.v-verdict { border-radius:14px; padding:.9rem 1.1rem; display:flex; gap:.8rem; align-items:flex-start; margin:1rem 0 .4rem; }\n.v-verdict.warn { background:var(--warn-soft); border:1px solid #F0D9B5; }\n.v-verdict.good { background:var(--good-soft); border:1px solid #CFE3D7; }\n.v-verdict.neutral { background:var(--navy-soft); border:1px solid #D6DFEC; }\n.v-verdict b { color:var(--ink); }\n.v-verdict p { margin:.15rem 0 0; color:var(--muted); font-size:.86rem; }\n\n/* Drivers */\n.v-driver { display:flex; justify-content:space-between; align-items:center; padding:.6rem .8rem;\n  border:1px solid var(--line); border-radius:12px; margin-bottom:.45rem; background:var(--card); }\n.v-driver .l { font-size:.88rem; color:var(--ink); } .v-driver .c { font-size:.74rem; color:var(--muted); }\n.v-driver .d { font-weight:600; font-size:.92rem; }\n.v-driver .d.pos { color:var(--good); } .v-driver .d.neg { color:var(--accent); }\n\n/* Pipeline */\n.v-pipe { display:grid; grid-template-columns: 1.3fr 1.3fr .9fr 2fr; gap:.4rem .8rem; align-items:center;\n  padding:.6rem .2rem; border-bottom:1px solid var(--line); font-size:.85rem; }\n.v-pipe:last-child { border-bottom:none; }\n.v-pipe .arrow { color:var(--muted); }\n.v-status { font-size:.72rem; font-weight:600; padding:.18rem .55rem; border-radius:999px; }\n.s-completed { background:var(--good-soft); color:var(--good); }\n.s-skipped, .s-pending { background:var(--grey-soft); color:var(--muted); }\n.s-failed { background:var(--accent-soft); color:var(--accent); }\n.s-replayed { background:var(--warn-soft); color:var(--warn); }\n.v-pipe .det { color:var(--muted); font-size:.8rem; }\n\n/* Comparables */\n.v-comp .price { font-size:1.15rem; font-weight:600; color:var(--ink); }\n.v-comp .row { display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); margin-top:.3rem; }\n\n/* Report paper */\n.v-paper { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.6rem 1.9rem;\n  box-shadow: 0 10px 30px rgba(20,33,61,.06); }\n\n/* Notices */\n.v-note { border-left:3px solid var(--accent); background:var(--card); padding:.7rem .95rem; border-radius:0 10px 10px 0;\n  font-size:.86rem; color:var(--muted); margin:.6rem 0 1rem; }\n.v-note.demo { border-left-color:var(--warn); background:var(--warn-soft); color:#6E4A12; }\n\n/* Widgets */\n.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; padding:.55rem 1.1rem;\n  transition: all .15s ease; }\n.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {\n  background:var(--ink); border:1px solid var(--ink); color:#fff; }\n.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {\n  background:#22335A; border-color:#22335A; transform:translateY(-1px); }\n.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {\n  background:var(--card); border:1px solid var(--line); color:var(--ink); }\n.stButton > button[kind="secondary"]:hover { border-color:var(--ink); color:var(--ink); }\n.stButton > button[kind="tertiary"] { color:var(--muted); }\n[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,\n[data-baseweb="select"] > div { border-radius:10px !important; background:#fff !important; }\n[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #CFC8BC; border-radius:16px; background:#FCFBF9;\n  padding:2.2rem 1rem; }\n[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--ink); }\n[data-testid="stExpander"] details { border:1px solid var(--line); border-radius:14px; background:var(--card); }\n[data-testid="stImage"] img { border-radius:12px; object-fit:cover; }\n.v-footer { margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted);\n  font-size:.76rem; display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }\n</style>\n'

def STY_inject() -> None:
    st.markdown(STY_CSS, unsafe_allow_html=True)


# ============================================================================
# ui/state.py: Session state: everything the workflow remembers between reruns and steps.
# ============================================================================
S_REVIEW_NUMERIC = ['area_m2', 'rooms', 'baths', 'floor', 'year_built']

def S_defaults() -> dict:
    return {'step': 1, 'f_text': '', 'f_address': '', 'f_hood': 'Sol (centre)', 'f_asking': 0, 'f_area': None, 'f_rooms': None, 'f_baths': None, 'f_lang': 'English', 'photos': [], 'uploader_n': 0, 'photo_errors': [], 'c_extract': None, 'c_photo': None, 'analysis': None, 'valuation': None, 'report': None, 'stages': E_empty_stages(), 'error': None, 'replay': None, 'flash': None, 'geo_cache': {}}

def S_init() -> None:
    ss = st.session_state
    for key, val in S_defaults().items():
        if key not in ss:
            ss[key] = val
    for key in list(ss.keys()):
        if isinstance(key, str) and key.startswith(('f_', 'r_')):
            ss[key] = ss[key]

def S_reset() -> None:
    """Start a new valuation: forget everything, then restore the defaults."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    S_init()

def S_go(step: int) -> None:
    st.session_state.step = step
    st.session_state.error = None

def S_overrides() -> dict:
    ss = st.session_state
    return {'area_m2': ss.f_area, 'rooms': ss.f_rooms, 'baths': ss.f_baths}

def S_extract_fp() -> str:
    return E_fingerprint('extract', st.session_state.f_text.strip(), USE_LLM)

def S_photo_fp() -> str:
    return E_fingerprint('photo', *[p['bytes'] for p in st.session_state.photos[:4]])

def S_analysis_fp() -> str:
    ss = st.session_state
    return E_fingerprint('analysis', S_extract_fp(), S_photo_fp(), S_overrides(), ss.f_asking, ss.f_address.strip(), ss.f_hood)

def S_review_features() -> dict:
    """Current values of the step-3 review widgets, applied on top of the analysis."""
    ss = st.session_state
    feats = dict(ss.analysis['features'])
    for key in S_REVIEW_NUMERIC:
        feats[key] = ss.get(f'r_{key}')
    feats['condition'] = ss.get('r_condition', feats.get('condition'))
    for key in E_AMENITIES:
        feats[key] = int(bool(ss.get(f'r_{key}', feats.get(key))))
    return feats
S_REVIEW_LIMITS = {'area_m2': (0, 1000), 'rooms': (0, 15), 'baths': (0, 10), 'floor': (-1, 40), 'year_built': (1500, 2030)}

def S_seed_review(feats: dict) -> None:
    ss = st.session_state
    for key in S_REVIEW_NUMERIC:
        val = feats.get(key)
        try:
            lo, hi = S_REVIEW_LIMITS[key]
            val = None if val in (None, '') else min(max(float(val), lo), hi)
        except (TypeError, ValueError):
            val = None
        ss[f'r_{key}'] = None if val is None else val if key == 'area_m2' else int(val)
    ss['r_condition'] = feats.get('condition') if feats.get('condition') in E_CONDITIONS else 'unknown'
    for key in E_AMENITIES:
        ss[f'r_{key}'] = bool(feats.get(key))

def S_provenance() -> dict:
    """Original source of each field, upgraded to 'edited' where the user changed it in step 3."""
    ss = st.session_state
    prov = dict(ss.analysis['prov'])
    original, current = (ss.analysis['features'], S_review_features())
    for key in S_REVIEW_NUMERIC + ['condition'] + list(E_AMENITIES):
        a, b = (original.get(key), current.get(key))
        if key in E_AMENITIES:
            a, b = (int(bool(a)), int(bool(b)))
        if key in S_REVIEW_NUMERIC and a is not None and (b is not None):
            a, b = (float(a), float(b))
        if a != b:
            prov[key] = 'edited'
    return prov

def S_valuation_fp(prepared: dict) -> str:
    keep = {k: v for k, v in prepared.items() if k not in ('highlights', 'address')}
    return E_fingerprint('valuation', keep, MARKET_ADJUSTMENT)

def S_report_fp() -> str:
    ss = st.session_state
    return E_fingerprint('report', ss.valuation['fp'], ss.f_lang, USE_LLM)


# ============================================================================
# ui/components.py: Reusable pieces of the ValorIA interface (HTML snippets + small Streamlit blocks).
# ============================================================================
C_STEPS = ['Property details', 'Photos', 'AI analysis', 'Valuation', 'Report']

def C_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)

def C_signed(x: float) -> str:
    return ('+' if x >= 0 else '−') + eur(abs(x))

def C_nav(on_reset) -> None:
    ss = st.session_state
    if ss.replay:
        dot, text = ('demo', f"Demo replay · saved {ss.replay['saved_at']}")
    elif USE_LLM:
        dot, text = ('', E_mode_label())
    else:
        dot, text = ('off', E_mode_label())
    left, right = st.columns([5, 1.25], vertical_alignment='center')
    with left:
        C_html(f'<div class="v-nav"><div class="v-logo">V</div>\n        <div><div class="v-brand">Valor<span>IA</span></div>\n        <div class="v-tagline">AI-powered property intelligence for Madrid</div></div>\n        <div style="margin-left:auto"><span class="v-pill"><span class="v-dot {dot}"></span>{escape(text)}</span></div>\n        </div>')
    with right:
        st.button('↺  New valuation', on_click=on_reset, width='stretch', key='nav_reset')

def C_stepper(current: int) -> None:
    items = []
    for i, name in enumerate(C_STEPS, start=1):
        cls = 'done' if i < current else 'active' if i == current else ''
        mark = '✓' if i < current else str(i)
        bar = '<div class="bar"></div>' if i < len(C_STEPS) else ''
        items.append(f'<div class="v-step {cls}"><div class="n">{mark}</div><div class="t">{name}</div>{bar}</div>')
    C_html(f"""<div class="v-stepper">{''.join(items)}</div>""")

def C_header(step: int, title: str, sub: str) -> None:
    C_html(f'<div class="v-eyebrow">Step {step} of {len(C_STEPS)}</div><div class="v-title">{escape(title)}</div><div class="v-sub">{escape(sub)}</div>')

def C_note(text: str, kind: str='') -> None:
    C_html(f'<div class="v-note {kind}">{text}</div>')

def C_badge(source: str) -> str:
    return f'<span class="v-badge b-{source}">{E_SOURCE_LABELS.get(source, source)}</span>'

def C_feature_card(icon: str, label: str, value, source: str, hint: str='') -> None:
    missing = value in (None, '', 'Unknown')
    shown = escape(hint) if missing and hint else 'Not found' if missing else escape(str(value))
    C_html(f"""<div class="v-card tight"><div class="v-feat-label">{icon} {escape(label)}</div>\n    <div class="v-feat-value {('missing' if missing else '')}">{shown}</div>{C_badge(source)}</div>""")

def C_replay_banner() -> None:
    r = st.session_state.replay
    if r and any((s['status'] == 'replayed' for s in st.session_state.stages.values())):
        C_note(f"<b>Demo replay.</b> These AI outputs were saved from a real run on {escape(r['saved_at'])} ({escape(r['mode'])}). They are being shown again, not recomputed live. Change any input to run that step live.", 'demo')

def C_flash() -> None:
    msg = st.session_state.flash
    if msg:
        kind, text = msg
        (st.warning if kind == 'warn' else st.success)(text)
        st.session_state.flash = None

def C_nav_buttons(back_step: int | None, primary_label: str, primary_key: str, extra=None):
    """Bottom navigation: Back on the left, primary action on the right. Returns True if primary clicked."""
    st.write('')
    cols = st.columns([1.1, 2.3, 1.2, 1.6] if extra else [1.1, 3.5, 1.6], vertical_alignment='center')
    if back_step:
        if cols[0].button('← Back', key=f'back_{primary_key}', width='stretch'):
            st.session_state.step = back_step
            st.session_state.error = None
            st.rerun()
    if extra:
        extra(cols[2])
    return cols[-1].button(primary_label, type='primary', key=primary_key, width='stretch')

def C_pipeline_panel(show_outputs: bool=True) -> None:
    ss = st.session_state
    with st.expander('How ValorIA calculated this valuation'):
        st.caption('Generative AI turns unstructured text and photos into structured data and explains the result. Traditional machine learning calculates the price. Each row shows what really ran.')
        rows = []
        for key, src, step, kind in E_STAGES:
            rec = ss.stages.get(key, {'status': 'pending'})
            tag = 'b-ml' if kind == 'Machine learning' else 'b-genai' if kind == 'GenAI' else 'b-listing'
            secs = f" · {rec['seconds']:.1f} s" if rec.get('seconds') else ''
            rows.append(f"""<div class="v-pipe"><div>{src} <span class="arrow">→</span></div>\n              <div><b>{step}</b><br><span class="v-badge {tag}">{kind}</span></div>\n              <div><span class="v-status s-{rec['status']}">{rec['status'].capitalize()}</span></div>\n              <div class="det">{escape(rec.get('detail') or '')}{secs}</div></div>""")
        C_html(f"""<div class="v-card">{''.join(rows)}</div>""")
        if show_outputs and ss.analysis:
            a, b = st.columns(2)
            with a:
                st.markdown('**Extracted features (raw output)**')
                raw = ss.c_extract['features'] if ss.c_extract else {}
                st.json({k: v for k, v in raw.items() if v not in (0, None, [], '')}, expanded=False)
            with b:
                st.markdown('**Photo analysis (raw output)**')
                if ss.analysis.get('photo'):
                    st.json(ss.analysis['photo'], expanded=False)
                else:
                    st.caption(ss.stages['vision'].get('detail') or 'Not run')

def C_price_bar(v: dict, asking: float | None) -> None:
    import altair as alt
    lo = min(v['low'], asking or v['low']) * 0.93
    hi = max(v['high'], asking or v['high']) * 1.07
    x = alt.X('x:Q', title=None, scale=alt.Scale(domain=[lo, hi]), axis=alt.Axis(format=',.0f', labelColor='#6B6F76', grid=False))
    band = alt.Chart(pd.DataFrame({'x': [v['low']], 'x2': [v['high']]})).mark_bar(height=18, cornerRadius=9, color='#E6E1D8').encode(x=x, x2='x2:Q')
    marks = pd.DataFrame({'x': [v['estimate']], 'what': ['Model estimate']})
    if asking:
        marks = pd.concat([marks, pd.DataFrame({'x': [asking], 'what': ['Asking price']})])
    ticks = alt.Chart(marks).mark_tick(thickness=4, size=34).encode(x=x, color=alt.Color('what:N', scale=alt.Scale(domain=['Model estimate', 'Asking price'], range=['#14213D', '#7D2E3A']), legend=alt.Legend(title=None, orient='top')), tooltip=[alt.Tooltip('what:N', title=' '), alt.Tooltip('x:Q', title='€', format=',.0f')])
    st.altair_chart((band + ticks).properties(height=95).configure_view(strokeWidth=0))
    st.caption("Grey band: spread of the random forest's trees (10th–90th percentile). It shows how much the trees disagree; it is not a calibrated confidence interval.")

def C_driver(d: dict) -> None:
    cls = 'pos' if d['delta'] >= 0 else 'neg'
    C_html(f"""<div class="v-driver"><div><div class="l">{escape(d['label'])}</div>\n    <div class="c">vs. {escape(d['compared'])}</div></div><div class="d {cls}">{C_signed(d['delta'])}</div></div>""")

def C_comp_card(c: dict, i: int, subject_ppm2: float) -> None:
    diff = c['price_per_m2'] / subject_ppm2 - 1 if subject_ppm2 else 0
    extras = ' · '.join((x for x in ['Lift' if c['has_lift'] else '', 'Terrace' if c['has_terrace'] else ''] if x))
    C_html(f"""<div class="v-card tight v-comp"><div class="v-feat-label">Comparable {i} · near {escape(c['near'])}</div>\n    <div class="price">{eur(c['price'])}</div>\n    <div class="row"><span>{c['area_m2']:.0f} m² · {c['rooms']} bed · {c['baths']} bath</span></div>\n    <div class="row"><span>{eur(c['price_per_m2'])}/m²</span><span>{diff:+.0%} vs. yours</span></div>\n    <div class="row"><span>{c['distance_km']:.2f} km away</span><span>{escape(extras) or '&nbsp;'}</span></div></div>""")

def C_footer() -> None:
    C_html('<div class="v-footer"><div>ValorIA provides automated estimates for information only.\n    They are not an official appraisal (tasación).</div>\n    <div>Training data: idealista18 (2018, ODbL 1.0) · University AI &amp; ML project</div></div>')


# ============================================================================
# ui/actions.py: What happens when the user presses a main button. Each action runs only the
# ============================================================================
def A__stage(key: str, record: dict) -> None:
    stages = dict(st.session_state.stages)
    stages[key] = record
    st.session_state.stages = stages

def A__pending(*keys: str) -> None:
    for key in keys:
        A__stage(key, {'status': 'pending', 'detail': '', 'seconds': None})

def A_extract_key(force_keywords: bool) -> str:
    return E_fingerprint(S_extract_fp(), 'keywords' if force_keywords else 'auto')

def A_analysis_key(force_keywords: bool) -> str:
    return E_fingerprint(S_analysis_fp(), A_extract_key(force_keywords))

def A__secs(rec: dict) -> str:
    return f" ({rec['seconds']:.1f} s)" if rec.get('seconds') else ''

def A_run_analysis(force_keywords: bool=False) -> bool:
    ss = st.session_state
    ss.error = None
    with st.status('Analysing the property…', expanded=True) as status:
        key = A_extract_key(force_keywords)
        if ss.c_extract and ss.c_extract['fp'] == key:
            st.write('✓ Property characteristics already extracted (inputs unchanged)')
        else:
            st.write('Extracting property characteristics…')
            try:
                feats, rec = E_analyse_listing(ss.f_text, use_keywords=force_keywords)
            except Exception as exc:
                msg = E_friendly_error(exc)
                A__stage('extract', {'status': 'failed', 'detail': msg, 'seconds': None})
                ss.error = {'stage': 'extract', 'msg': msg}
                status.update(label='Analysis stopped', state='error')
                return False
            ss.c_extract = {'fp': key, 'features': feats, 'record': rec, 'forced': force_keywords}
            st.write(f"✓ Characteristics extracted · {rec['detail']}{A__secs(rec)}")
        A__stage('extract', ss.c_extract['record'])
        pfp = S_photo_fp()
        if not ss.photos:
            ss.c_photo = {'fp': pfp, 'photo': None, 'record': E__record('skipped', 'No photos uploaded')}
        elif ss.c_photo and ss.c_photo['fp'] == pfp:
            st.write('✓ Photos already assessed (unchanged)')
        else:
            if USE_LLM:
                st.write('Assessing photographs…')
            try:
                photo, rec = E_analyse_photos([p['bytes'] for p in ss.photos])
                ss.c_photo = {'fp': pfp, 'photo': photo, 'record': rec}
                if rec['status'] == 'completed':
                    st.write(f"✓ Photos assessed · {rec['detail']}{A__secs(rec)}")
            except Exception as exc:
                msg = E_friendly_error(exc)
                ss.c_photo = {'fp': None, 'photo': None, 'record': E__record('failed', msg)}
                ss.flash = ('warn', f'Photo analysis failed: {msg} The valuation can continue without it.')
                st.write('⚠ Photo analysis failed, continuing without it')
        A__stage('vision', ss.c_photo['record'])
        loc_key = (ss.f_address.strip(), ss.f_hood)
        if loc_key not in ss.geo_cache:
            if loc_key[0]:
                st.write('Locating the address…')
            ss.geo_cache[loc_key] = E_locate(*loc_key)
        location = ss.geo_cache[loc_key]
        akey = A_analysis_key(force_keywords)
        if not ss.analysis or ss.analysis['fp'] != akey:
            feats, prov = E_merge_inputs(ss.c_extract['features'], S_overrides(), ss.c_photo['photo'], ss.f_asking)
            ss.analysis = {'fp': akey, 'features': feats, 'prov': prov, 'photo': ss.c_photo['photo'], 'location': location}
            S_seed_review(feats)
            A__pending('predict', 'comps', 'report')
        status.update(label='Analysis complete', state='complete', expanded=False)
    return True

def A_retry_photos() -> None:
    st.session_state.c_photo = None
    st.session_state.analysis = None

def A_validate_review() -> str | None:
    ss = st.session_state
    if not ss.get('r_area_m2') or ss['r_area_m2'] < 15:
        return 'Please enter the property size (at least 15 m²). The pricing model needs it.'
    if ss.get('r_rooms') is None:
        return 'Please enter the number of bedrooms (0 for a studio).'
    return None

def A_prepared_features() -> dict:
    loc = st.session_state.analysis['location']
    return E_prepare(S_review_features(), loc['lat'], loc['lon'])

def A_run_valuation() -> bool:
    ss = st.session_state
    ss.error = None
    prepared = A_prepared_features()
    vfp = S_valuation_fp(prepared)
    if ss.valuation and ss.valuation['fp'] == vfp:
        return True
    with st.status('Valuing the property…', expanded=True) as status:
        st.write('Preparing valuation features…')
        try:
            st.write('Running the pricing model…')
            v, rec = E_predict(prepared)
            st.write(f'✓ Price predicted by the random forest{A__secs(rec)}')
            st.write('Measuring which characteristics move the price…')
            drivers = E_sensitivity(prepared, v['estimate'])
        except Exception as exc:
            msg = E_friendly_error(exc)
            A__stage('predict', {'status': 'failed', 'detail': msg, 'seconds': None})
            ss.error = {'stage': 'predict', 'msg': msg}
            status.update(label='Valuation failed', state='error')
            return False
        ss.valuation = {'fp': vfp, 'features': prepared, 'valuation': v, 'drivers': drivers, 'verdict': E_verdict(prepared.get('asking_price'), v), 'prov': S_provenance(), 'comps': None}
        A__stage('predict', rec)
        A__pending('comps', 'report')
        status.update(label='Valuation ready', state='complete', expanded=False)
    return True

def A_context() -> dict:
    ss = st.session_state
    val = ss.valuation
    return {'features': val['features'], 'valuation': val['valuation'], 'verdict': val['verdict'], 'drivers': val['drivers'], 'comps': val.get('comps') or [], 'photo': ss.analysis['photo'], 'location': ss.analysis['location']}

def A_run_report(force: bool=False) -> None:
    ss = st.session_state
    rfp = S_report_fp()
    if not force and ss.report and (ss.report['fp'] == rfp):
        return
    with st.status('Preparing your report…', expanded=True) as status:
        if ss.valuation.get('comps') is None:
            st.write('Retrieving comparable properties…')
            try:
                comps, rec = E_comparables(ss.valuation['features'])
                st.write(f'✓ {len(comps)} comparable flats retrieved{A__secs(rec)}')
            except Exception as exc:
                comps, rec = ([], E__record('failed', E_friendly_error(exc)))
                st.write('⚠ Comparable retrieval failed')
            ss.valuation = {**ss.valuation, 'comps': comps}
            A__stage('comps', rec)
        ctx = A_context()
        if USE_LLM:
            st.write('Writing the valuation report…')
        try:
            md, rec = E_write_client_report(ctx, ss.f_lang)
        except Exception as exc:
            md = E_template_report(ctx)
            rec = E__record('failed', f'{E_friendly_error(exc)} A template report was used instead.')
        A__stage('report', rec)
        st.write('Building the PDF…')
        ss.report = {'fp': rfp, 'md': md, 'pdf': P_build_pdf(ctx, md, eur), 'record': rec}
        status.update(label='Report ready', state='complete', expanded=False)

def A_save_snapshot(name: str) -> str:
    ss = st.session_state
    state = {'inputs': {k: ss[k] for k in ss.keys() if isinstance(k, str) and k.startswith('f_')}, 'photos': [p['bytes'] for p in ss.photos], 'c_extract': ss.c_extract, 'c_photo': {k: v for k, v in ss.c_photo.items() if k != 'fp'}, 'analysis': {k: v for k, v in ss.analysis.items() if k != 'fp'}, 'valuation': {k: v for k, v in ss.valuation.items() if k != 'fp'}, 'report': {'md': ss.report['md'], 'record': ss.report['record']}, 'review': S_review_features(), 'stages': ss.stages}
    return E_save_snapshot(name, state)

def A_load_snapshot(stem: str) -> None:
    data = E_load_snapshot(stem)
    S_reset()
    S_init()
    ss = st.session_state
    when = data.get('saved_at', '?').replace('T', ' ')

    def replayed(rec: dict) -> dict:
        if rec and rec.get('status') in ('completed', 'failed'):
            return {'status': 'replayed', 'detail': f"{rec['detail']} · saved run {when}", 'seconds': None}
        return rec
    for k, v in data['inputs'].items():
        ss[k] = v
    ss.photos = [{'id': E_fingerprint(b), 'name': f'Photo {i + 1}', 'bytes': b} for i, b in enumerate(data['photos'])]
    forced = bool(data['c_extract'].get('forced'))
    ss.c_extract = {**data['c_extract'], 'fp': A_extract_key(forced), 'record': replayed(data['c_extract']['record'])}
    ss.c_photo = {**data['c_photo'], 'fp': S_photo_fp(), 'record': replayed(data['c_photo']['record'])}
    ss.analysis = {**data['analysis'], 'fp': A_analysis_key(forced)}
    ss.geo_cache = {(ss.f_address.strip(), ss.f_hood): ss.analysis['location']}
    S_seed_review(data['review'])
    ss.valuation = {**data['valuation'], 'fp': S_valuation_fp(A_prepared_features())}
    ctx = A_context()
    ss.report = {'fp': S_report_fp(), 'md': data['report']['md'], 'pdf': P_build_pdf(ctx, data['report']['md'], eur), 'record': replayed(data['report']['record'])}
    ss.stages = {k: replayed(v) for k, v in data['stages'].items()}
    ss.replay = {'name': stem, 'saved_at': when, 'mode': data.get('mode', '?'), 'market_adjustment': data.get('market_adjustment')}
    ss.step = 1


# ============================================================================
# ui/step_details.py: Step 1 - Tell us about the property.
# ============================================================================
DET_MIN_TEXT = 40

def DET__use_example(key: str) -> None:
    ex = E_load_example(key)
    ss = st.session_state
    ss.f_text, ss.f_hood, ss.f_asking, ss.f_address = (ex['text'], ex['hood'], ex['asking'], '')
    ss.f_area = ss.f_rooms = ss.f_baths = None
    ss.flash = ('ok', f"Example loaded: {ex['title']}. This is an invented listing for demonstration.")

def DET__use_saved(label: str) -> None:
    ss = st.session_state
    d = E_load_saved_listing(E_saved_listings()[label])
    ss.f_text, ss.f_asking, ss.f_address = (d['text'], d['asking'], d['address'])
    ss.f_hood = d['hood'] or ss.f_hood
    ss.f_area = ss.f_rooms = ss.f_baths = None
    ss.photos = [{'id': E_fingerprint(b), 'name': n, 'bytes': b} for b, n in zip(d['photos'], d['photo_names'])]
    ss.uploader_n += 1
    if d['coords']:
        ss.geo_cache[d['address'].strip(), ss.f_hood] = {'lat': d['coords'][0], 'lon': d['coords'][1], 'label': d['neighbourhood'] or ss.f_hood, 'source': 'saved location (approximate, from the team)', 'warning': None}
    ss.flash = ('ok', f'Loaded: {label}' + (f" with {len(d['photos'])} photo(s)." if d['photos'] else '.'))

def DET_render() -> None:
    C_header(1, 'Tell us about the property', 'Start with the listing information. ValorIA will identify the characteristics that influence its market value.')
    C_flash()
    st.markdown('**Try an example property**')
    cols = st.columns(len(E_EXAMPLES))
    for col, (key, ex) in zip(cols, E_EXAMPLES.items()):
        with col:
            C_html(f"""<div class="v-card tight"><h4>{ex['title']}</h4><p>{ex['blurb']}</p><p style="margin-top:.35rem"><span class="v-badge b-missing">Invented example</span></p></div>""")
            st.button('Use this example', key=f'ex_{key}', on_click=DET__use_example, args=(key,), width='stretch')
    saved = E_saved_listings()
    if saved:
        st.markdown('**Real listings saved by the team**')
        cols = st.columns(min(len(saved), 4))
        for col, label in zip(cols, list(saved)[:4]):
            with col:
                C_html(f'<div class="v-card tight"><h4>{label}</h4><p style="margin-top:.35rem"><span class="v-badge b-listing">Real listing</span></p></div>')
                st.button('Use this listing', key=f'saved_{label}', on_click=DET__use_saved, args=(label,), width='stretch')
    snapshots = E_list_snapshots()
    if snapshots:
        with st.expander('Presentation backup: replay a saved run'):
            st.caption('Loads the outputs of a real, previously completed run. Use it only if the internet or the AI service fails during a demo. The app labels everything as a replay.')
            a, b = st.columns([3, 1], vertical_alignment='bottom')
            pick = a.selectbox('Saved run', snapshots, key='snap_pick')
            b.button('Load saved run', on_click=A_load_snapshot, args=(pick,), width='stretch')
    st.write('')
    left, right = st.columns([1.6, 1], gap='large')
    with left:
        st.text_area('Property description (Spanish or English)', key='f_text', height=255, placeholder="Paste the listing text, e.g. 'Piso exterior en Chamberí, 85 m², 3 habitaciones, 2 baños, cuarta planta con ascensor…'")
        n = len(st.session_state.f_text.strip())
        st.caption(f'{n} characters' + ('' if n >= DET_MIN_TEXT or n == 0 else f' · at least {DET_MIN_TEXT} needed'))
    with right:
        st.text_input('Madrid address (optional)', key='f_address', placeholder='e.g. Calle de Fuencarral 100')
        st.selectbox('Neighbourhood (used if no address is given)', list(E_NEIGHBOURHOODS), key='f_hood')
        st.number_input('Asking price in € (optional)', min_value=0, step=5000, key='f_asking', help='Leave at 0 to value the property without an asking price.')
        with st.expander('Know the exact figures? (optional)'):
            st.caption('Values entered here take priority over what the AI reads from the description.')
            a, b, c = st.columns(3)
            a.number_input('Size m²', min_value=0.0, max_value=1000.0, step=1.0, value=None, key='f_area')
            b.number_input('Bedrooms', min_value=0, max_value=15, step=1, value=None, key='f_rooms')
            c.number_input('Baths', min_value=0, max_value=10, step=1, value=None, key='f_baths')
    if C_nav_buttons(None, 'Continue to property photos →', 'to_photos'):
        if len(st.session_state.f_text.strip()) < DET_MIN_TEXT:
            st.error(f'Please add a property description of at least {DET_MIN_TEXT} characters, or load an example.')
        else:
            st.session_state.step = 2
            st.rerun()


# ============================================================================
# ui/step_photos.py: Step 2 - Show us the property.
# ============================================================================
PHO_MAX_PHOTOS = 8
PHO_VISION_LIMIT = 4

def PHO__valid_image(data: bytes) -> bool:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return im.format in ('JPEG', 'PNG')
    except Exception:
        return False

def PHO__add_uploads(files) -> None:
    ss = st.session_state
    known = {p['id'] for p in ss.photos}
    errors = []
    for f in files:
        data = f.getvalue()
        pid = E_fingerprint(data)
        if pid in known:
            continue
        if not PHO__valid_image(data):
            errors.append(f'{f.name} is not a valid JPG or PNG image and was skipped.')
        elif len(ss.photos) >= PHO_MAX_PHOTOS:
            errors.append(f'Only {PHO_MAX_PHOTOS} photos can be added; {f.name} was skipped.')
        else:
            ss.photos = ss.photos + [{'id': pid, 'name': f.name, 'bytes': data}]
            known.add(pid)
    ss.photo_errors = errors
    ss.uploader_n += 1

def PHO__remove(pid: str) -> None:
    st.session_state.photos = [p for p in st.session_state.photos if p['id'] != pid]

def PHO__move_first(pid: str) -> None:
    ps = st.session_state.photos
    st.session_state.photos = [p for p in ps if p['id'] == pid] + [p for p in ps if p['id'] != pid]

def PHO_render() -> None:
    ss = st.session_state
    C_header(2, 'Show us the property', "Upload photographs to help our AI understand the property's condition, finishes, and overall quality.")
    C_flash()
    if not USE_LLM:
        C_note('<b>Offline mode.</b> Without an API key there is no vision model, so photos will be kept with the valuation but <b>not analysed</b>.')
    files = st.file_uploader('Drop photos here (JPG, JPEG or PNG)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key=f'uploader_{ss.uploader_n}', label_visibility='collapsed')
    if files:
        PHO__add_uploads(files)
        st.rerun()
    for err in ss.photo_errors:
        st.warning(err)
    if ss.photos:
        n = len(ss.photos)
        analysed = min(n, PHO_VISION_LIMIT)
        st.markdown(f"**{n} photo{('s' if n != 1 else '')} added** · {('the first ' + str(analysed) + ' are' if n > PHO_VISION_LIMIT else 'all are')} sent to the vision model. Kitchens and bathrooms tell the model the most.")
        for row in range(0, n, 4):
            cols = st.columns(4)
            for col, (i, p) in zip(cols, list(enumerate(ss.photos))[row:row + 4]):
                with col:
                    st.image(p['bytes'])
                    label = 'Analysed' if i < PHO_VISION_LIMIT and USE_LLM else 'Not analysed'
                    st.caption(f"{p['name'][:26]} · {label}")
                    a, b = st.columns(2)
                    a.button('Remove', key=f"rm_{p['id']}", on_click=PHO__remove, args=(p['id'],), type='tertiary')
                    if i >= PHO_VISION_LIMIT:
                        b.button('Use first', key=f"mv_{p['id']}", on_click=PHO__move_first, args=(p['id'],), type='tertiary')
    else:
        st.caption("No photos? That's fine: you can continue, and the valuation will rely on the description.")
    err = ss.error
    if err and err['stage'] == 'extract':
        st.error(f"**The listing could not be analysed.** {err['msg']}")
        a, b, _ = st.columns([1.2, 2, 2])
        if a.button('Try again', type='primary'):
            if A_run_analysis():
                ss.step = 3
            st.rerun()
        if USE_LLM and b.button('Use offline keyword extraction instead'):
            if A_run_analysis(force_keywords=True):
                ss.flash = ('warn', 'Features were read with simple keyword rules instead of the LLM. Please check them carefully below.')
                ss.step = 3
            st.rerun()
    label = 'Analyse property →' if ss.photos else 'Skip photos and analyse →'
    if C_nav_buttons(1, label, 'analyse'):
        if A_run_analysis():
            ss.step = 3
        st.rerun()


# ============================================================================
# ui/step_analysis.py: Step 3 - Understanding your property: what the AI extracted, with sources, editable.
# ============================================================================
def ANA__fmt(key: str, val):
    if val in (None, ''):
        return None
    if key == 'area_m2':
        return f'{float(val):.0f} m²'
    if key == 'floor':
        return 'Ground floor' if int(val) == 0 else f'Floor {int(val)}'
    return str(int(val))

def ANA__score_bar(label: str, score) -> str:
    if score in (None, ''):
        return f'<div class="v-driver"><div class="l">{label}</div><div class="c">Not visible</div></div>'
    pct = max(0, min(5, float(score))) / 5 * 100
    return f'<div style="margin:.35rem 0"><div style="display:flex;justify-content:space-between;font-size:.84rem"><span>{label}</span><b>{score}/5</b></div><div style="height:7px;background:#F0EEEA;border-radius:4px"><div style="width:{pct:.0f}%;height:7px;background:#5B4B8A;border-radius:4px"></div></div></div>'

def ANA__photo_card() -> None:
    ss = st.session_state
    photo, rec = (ss.analysis.get('photo'), ss.stages['vision'])
    if photo and (photo.get('condition') or photo.get('kitchen_score') or photo.get('notes')):
        cond = E_CONDITIONS.get(photo.get('condition'), 'Not rated')
        known = {'condition', 'kitchen_score', 'bathroom_score', 'finishes_score', 'notes'}
        other = ''.join((f"<p><b>{escape(str(k).replace('_', ' ').capitalize())}:</b> {escape(str(v))}</p>" for k, v in photo.items() if k not in known and v not in (None, '')))
        C_html(f"""<div class="v-card"><div class="v-feat-label">📷 Photo assessment {C_badge('photo')}</div>\n        <div class="v-feat-value">{escape(cond)}</div>\n        {ANA__score_bar('Kitchen', photo.get('kitchen_score'))}{ANA__score_bar('Bathroom', photo.get('bathroom_score'))}\n        {ANA__score_bar('Finishes', photo.get('finishes_score'))}\n        <p style="margin-top:.5rem">{escape(str(photo.get('notes') or ''))}</p>{other}\n        <p style="margin-top:.4rem;font-size:.75rem">{escape(rec.get('detail') or '')}</p></div>""")
    else:
        C_html(f"""<div class="v-card"><div class="v-feat-label">📷 Photo assessment</div>\n        <div class="v-feat-value missing">Not analysed</div><p>{escape(rec.get('detail') or 'No photos')}</p></div>""")
        if rec.get('status') == 'failed' and st.button('Retry photo analysis'):
            A_retry_photos()
            A_run_analysis()
            st.rerun()

def ANA_render() -> None:
    ss = st.session_state
    C_header(3, 'Understanding your property', 'Here is what ValorIA understood from your inputs. Check it, correct anything that is wrong, then calculate the valuation.')
    C_flash()
    C_note('ValorIA uses language and vision models to convert unstructured property information into features that can be processed by a machine learning valuation model.')
    loc = ss.analysis['location']
    if loc.get('warning'):
        st.warning(loc['warning'])
    feats, prov = (S_review_features(), S_provenance())
    C_html("<div style='margin:.2rem 0 .7rem;font-size:.8rem;color:#6B6F76'>Sources: " + ' '.join((C_badge(s) for s in ['user', 'listing', 'photo', 'edited', 'missing'])) + '</div>')
    row1 = st.columns(4)
    with row1[0]:
        C_html(f"""<div class="v-card tight"><div class="v-feat-label">📍 Location</div>\n        <div class="v-feat-value" style="font-size:1.1rem">{escape(loc['label'])}</div>{C_badge('user')}\n        <p style="font-size:.72rem;margin-top:.35rem">{escape(loc['source'])}</p></div>""")
    for col, key in zip(row1[1:], ['area_m2', 'rooms', 'baths']):
        with col:
            label, icon = E_NUMERIC_FIELDS[key]
            hint = 'Missing · estimated from 15 similar flats' if key in E_MODEL_DEFAULTS else 'Missing · required'
            C_feature_card(icon, label, ANA__fmt(key, feats.get(key)), prov.get(key, 'missing'), hint)
    row2 = st.columns(4)
    for col, key in zip(row2[:2], ['floor', 'year_built']):
        with col:
            label, icon = E_NUMERIC_FIELDS[key]
            C_feature_card(icon, label, ANA__fmt(key, feats.get(key)), prov.get(key, 'missing'), 'Missing · estimated from 15 similar flats')
    with row2[2]:
        cond = feats.get('condition')
        C_feature_card('🛠️', 'Condition', E_CONDITIONS.get(cond) if cond != 'unknown' else None, prov.get('condition', 'missing'), 'Unknown · treated as good')
    with row2[3]:
        asking = feats.get('asking_price')
        C_feature_card('🏷️', 'Asking price', eur(asking) if asking else None, prov.get('asking_price', 'missing'), 'Not provided')
    st.write('')
    left, right = st.columns([1.5, 1], gap='large')
    with left:
        st.markdown('**Amenities and characteristics**')
        on = [k for k in E_AMENITIES if feats.get(k)]
        chips = ''.join((f"""<span class="v-chip on">✓ {E_AMENITIES[k]} {C_badge(prov.get(k, 'listing'))}</span>""" for k in on))
        C_html(chips or "<p style='color:#6B6F76'>None mentioned in the listing.</p>")
        off = [E_AMENITIES[k] for k in E_AMENITIES if not feats.get(k)]
        st.caption('Not mentioned (treated as absent): ' + ', '.join(off))
        if feats.get('highlights'):
            items = ''.join((f'<li>{escape(str(h))}</li>' for h in feats['highlights']))
            C_html(f"""<div class="v-card tight"><div class="v-feat-label">✨ Highlights read by the AI {C_badge('listing')}</div><ul style="margin:.4rem 0 0 1rem;font-size:.86rem">{items}</ul></div>""")
    with right:
        ANA__photo_card()
    st.write('')
    with st.container(border=True):
        st.markdown('**Review and correct before valuing**')
        st.caption('Changes here are marked “Corrected by you” and are what the pricing model will use.')
        n = st.columns(5)
        n[0].number_input('Size (m²)', min_value=0.0, max_value=1000.0, step=1.0, value=None, key='r_area_m2')
        n[1].number_input('Bedrooms', min_value=0, max_value=15, step=1, value=None, key='r_rooms')
        n[2].number_input('Bathrooms', min_value=0, max_value=10, step=1, value=None, key='r_baths')
        n[3].number_input('Floor (0 = ground)', min_value=-1, max_value=40, step=1, value=None, key='r_floor')
        n[4].number_input('Year built', min_value=1500, max_value=2030, step=1, value=None, key='r_year_built')
        st.selectbox('Condition', list(E_CONDITIONS), format_func=E_CONDITIONS.get, key='r_condition')
        keys = list(E_AMENITIES)
        for start in range(0, len(keys), 5):
            cols = st.columns(5)
            for col, key in zip(cols, keys[start:start + 5]):
                col.toggle(E_AMENITIES[key], key=f'r_{key}')
    err = ss.error
    if err and err['stage'] == 'predict':
        st.error(f"**The valuation could not be calculated.** {err['msg']}")
    C_pipeline_panel()
    if C_nav_buttons(2, 'Calculate valuation →', 'value'):
        problem = A_validate_review()
        if problem:
            st.error(problem)
        else:
            if A_run_valuation():
                ss.step = 4
            st.rerun()


# ============================================================================
# ui/step_valuation.py: Step 4 - Your property valuation (the price comes from the ML model only).
# ============================================================================
def VAL__kpi(label: str, value: str, sub: str='') -> None:
    C_html(f'<div class="v-card tight v-kpi"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>')

def VAL__model_performance() -> None:
    with st.expander('Model performance (measured on held-out test data)'):
        m = E_load_metrics()
        if not m:
            st.info('No saved evaluation yet. Run `python -m src.train_model` once; it now saves the test-set scores to `models/metrics.json`, and they will appear here.')
            return
        rows = ''.join((f"<tr><td>{escape(name)}</td><td style='text-align:right'>{eur(s['mae'])}</td><td style='text-align:right'>{s['mape']:.1%}</td></tr>" for name, s in m['models'].items()))
        C_html(f"""<div class="v-card"><table style="width:100%;font-size:.88rem">\n        <tr style="color:#6B6F76"><td>Model</td><td style="text-align:right">Mean absolute error</td>\n        <td style="text-align:right">Mean % error</td></tr>{rows}\n        <tr><td>Enhanced model with AI-extracted features</td><td colspan="2" style="text-align:right;color:#6B6F76">\n        Not built yet</td></tr></table>\n        <p style="margin-top:.6rem">{m['n_test']:,} test flats · {escape(m['split'])} · trained {escape(m['trained_on'])}.</p>\n        <p>Both models use the structured idealista18 fields. At prediction time the LLM supplies the same\n        fields from free text. A model trained with extra AI-only features (e.g. photo scores) has not been\n        trained or evaluated yet, so no improvement can be claimed.</p></div>""")

def VAL_render() -> None:
    ss = st.session_state
    val = ss.valuation
    v, f, vd = (val['valuation'], val['features'], val['verdict'])
    info = E_model_size()
    C_header(4, 'Your property valuation', 'The price below is calculated by our machine learning model from the characteristics you confirmed.')
    trained = f"trained on {info['n_rows']:,} Madrid listings with {info['n_trees']} decision trees" if info else 'trained on Madrid listings'
    C_html(f"""<div class="v-hero"><div class="lbl">Estimated market value</div>\n    <div class="big">{eur(v['estimate'])}</div>\n    <div class="meta">Predicted by a random forest {trained} (idealista18, 2018 prices × market adjustment\n    {MARKET_ADJUSTMENT:g}). The language model did not set or change this number.</div></div>""")
    st.write('')
    k = st.columns(4)
    with k[0]:
        VAL__kpi('Price per m²', eur(v['per_m2']), f"{f.get('area_m2'):.0f} m²")
    with k[1]:
        VAL__kpi('Model tree spread', f"{eur(v['low'])[:-2]}–{eur(v['high'])}", '10th–90th percentile of trees')
    with k[2]:
        VAL__kpi('Asking price', eur(f['asking_price']) if vd else '—', 'provided' if vd else 'not provided')
    with k[3]:
        VAL__kpi('Difference', f"{vd['pct']:+.1%}" if vd else '—', C_signed(vd['eur']) if vd else 'needs an asking price')
    if vd:
        icon = {'warn': '▲', 'good': '▼', 'neutral': '≈'}[vd['tone']]
        extra = " It is also outside the spread of the model's trees." if vd['outside_range'] else " It is within the spread of the model's trees."
        C_html(f"""<div class="v-verdict {vd['tone']}"><div style="font-size:1.2rem">{icon}</div><div>\n        <b>{escape(vd['label'])}</b><p>The asking price is {vd['pct']:+.0%} versus the model estimate\n        (“aligned” means within ±{vd['tolerance']:.0%}).{extra} This is an indication, not investment advice.</p></div></div>""")
    if MARKET_ADJUSTMENT == 1.0:
        C_note("<b>Prices are at 2018 levels.</b> MARKET_ADJUSTMENT in <code>.env</code> is 1.0, so the estimate and comparables are not yet scaled to today's market. Set it from a sourced Madrid price index before comparing with current asking prices.")
    C_price_bar(v, f.get('asking_price'))
    st.markdown('#### Why this property is valued at this price')
    st.caption('Model sensitivity: how much the estimate changes when one characteristic is switched, with everything else unchanged. Measured by re-running the model, not estimated by the AI. Effects interact, so they do not add up to the total. Effects under 1% of the estimate are not shown: at that size they are within the model\'s noise.')
    pos = [d for d in val['drivers'] if d['delta'] >= 0]
    neg = [d for d in val['drivers'] if d['delta'] < 0]
    a, b = st.columns(2, gap='large')
    with a:
        st.markdown('**Adds value**')
        for d in pos[:6]:
            C_driver(d)
        if not pos:
            st.caption('No characteristic measurably adds value.')
    with b:
        st.markdown('**Reduces value**')
        for d in neg[:6]:
            C_driver(d)
        if not neg:
            st.caption('No characteristic measurably reduces value.')
    st.markdown('**Location context** · distances the model uses')
    lc = st.columns(3)
    for col, (label, key) in zip(lc, [('City centre (Sol)', 'dist_center_km'), ('Nearest metro', 'dist_metro_km'), ('Paseo de la Castellana', 'dist_castellana_km')]):
        with col:
            VAL__kpi(label, f'{f.get(key, 0):.2f} km', escape(ss.analysis['location']['source']))
    st.write('')
    VAL__model_performance()
    C_pipeline_panel(show_outputs=False)
    if C_nav_buttons(3, 'View comparables & report →', 'to_report'):
        A_run_report()
        ss.step = 5
        st.rerun()


# ============================================================================
# ui/step_report.py: Step 5 - Comparable properties and the client-ready report.
# ============================================================================
def REP__comparables() -> None:
    ss = st.session_state
    comps, v = (ss.valuation.get('comps') or [], ss.valuation['valuation'])
    st.markdown('#### Comparable properties')
    if not comps:
        st.info(f"No comparable flats could be retrieved. {ss.stages['comps'].get('detail', '')}")
        return
    med = float(np.median([c['price_per_m2'] for c in comps]))
    st.caption(f"{len(comps)} real flats from the idealista18 dataset, the nearest by location, size and bedrooms. Median {eur(med)}/m² against {eur(v['per_m2'])}/m² for this property. Prices are 2018 asking prices × {MARKET_ADJUSTMENT:g}. The dataset has no addresses, so the area shown is the nearest reference neighbourhood.")
    cols = st.columns(len(comps))
    for i, (col, c) in enumerate(zip(cols, comps), start=1):
        with col:
            C_comp_card(c, i, v['per_m2'])
    with st.expander('Map and table'):
        loc = ss.analysis['location']
        points = pd.DataFrame([{'lat': c['latitude'], 'lon': c['longitude'], 'color': '#B98A8F', 'size': 40} for c in comps] + [{'lat': loc['lat'], 'lon': loc['lon'], 'color': '#14213D', 'size': 70}])
        st.map(points, latitude='lat', longitude='lon', color='color', size='size', zoom=14)
        st.caption('Navy: this property · Rose: comparables')
        table = pd.DataFrame(comps)[['near', 'area_m2', 'rooms', 'baths', 'price', 'price_per_m2', 'distance_km']]
        st.dataframe(table.rename(columns={'near': 'Area', 'area_m2': 'm²', 'rooms': 'Bed', 'baths': 'Bath', 'price': 'Price €', 'price_per_m2': '€/m²', 'distance_km': 'km away'}).round(2), hide_index=True)

def REP__snapshot_panel() -> None:
    ss = st.session_state
    if ss.replay:
        return
    with st.expander('Presentation backup: save this run'):
        st.caption('Saves these real outputs (inputs, AI extraction, photo analysis, valuation, report) to `data/demo_runs/`. If the internet fails during the demo, load it from step 1. It will be clearly labelled as a replay.')
        a, b = st.columns([3, 1], vertical_alignment='bottom')
        name = a.text_input('Name', value='demo_' + ss.analysis['location']['label'], key='snap_name')
        if b.button('Save run', width='stretch'):
            saved = A_save_snapshot(name)
            st.success(f'Saved as {saved}')

def REP_render() -> None:
    ss = st.session_state
    if not ss.report or ss.report['fp'] != S_report_fp():
        A_run_report()
    C_header(5, 'Your valuation report', 'Comparable evidence and a client-ready summary, generated from the actual pipeline results.')
    REP__comparables()
    st.write('')
    st.markdown('#### Valuation report')
    rec = ss.report['record']
    meta_cols = st.columns([3, 1.3, 1], vertical_alignment='bottom')
    with meta_cols[0]:
        tag = {'completed': 'b-genai', 'replayed': 'b-edited', 'failed': 'b-edited'}.get(rec['status'], 'b-missing')
        C_html(f"""<span class="v-badge {tag}">{escape(rec.get('detail') or '')}</span>""")
    if USE_LLM:
        meta_cols[1].selectbox('Language', ['English', 'Spanish'], key='f_lang', label_visibility='collapsed')
        if meta_cols[2].button('Regenerate', width='stretch'):
            A_run_report(force=True)
            st.rerun()
    with st.container(border=True):
        st.markdown(ss.report['md'])
    st.write('')
    d = st.columns([1.4, 1.2, 1.2, 1.4])
    if ss.report.get('pdf'):
        d[0].download_button('⬇  Download valuation report', ss.report['pdf'], file_name='ValorIA_valuation.pdf', mime='application/pdf', type='primary', width='stretch')
    else:
        d[0].caption('PDF export needs `pip install reportlab`.')
    d[1].download_button('Markdown version', ss.report['md'], file_name='ValorIA_valuation.md', width='stretch')
    if d[2].button('← Back to valuation', width='stretch'):
        ss.step = 4
        st.rerun()
    d[3].button('Start a new valuation', on_click=S_reset, width='stretch')
    st.write('')
    REP__snapshot_panel()
    C_pipeline_panel()


# ============================================================================
# app.py: ValorIA: guided property valuation for Madrid.   Run:  streamlit run app.py
# ============================================================================
st.set_page_config(page_title='ValorIA · Property intelligence', page_icon='🏠', layout='wide', initial_sidebar_state='collapsed')
PAGES = {1: DET_render, 2: PHO_render, 3: ANA_render, 4: VAL_render, 5: REP_render}
STY_inject()
S_init()
ss = st.session_state
if ss.step >= 3 and (not ss.analysis):
    ss.step = 2 if ss.f_text.strip() else 1
if ss.step >= 4 and (not ss.valuation):
    ss.step = 3
C_nav(on_reset=S_reset)
C_stepper(ss.step)
C_replay_banner()
PAGES[ss.step]()
C_footer()
