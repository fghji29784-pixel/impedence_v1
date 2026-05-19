# DCIM Battery Analyzer

## Structure

- `app.py`: Streamlit entry point and analysis pipeline.
- `sidebar.py`: Sidebar widgets. Functions return values only.
- `loader.py`: BioLogic/CSV/TXT loading for charge and EIS data.
- `preprocessor.py`: p0/p1/p2 detection, Rs, time-window slicing, joint/relaxation data extraction.
- `models.py`: DCIM equivalent-circuit responses, Nyquist impedance, and parameter fitting.
- `eis_fitter.py`: EIS CNLS fitting for 2RC, 2RC_CPE, 3RC, and Randles_W.
- `plotter.py`: Matplotlib plots for raw data, fit results, Nyquist, and EIS fits.
- `views.py`: Streamlit tab renderers. Render functions take no arguments and read `st.session_state`.
- `diagnostics.py`: SOH/wetness/self-discharge diagnostics.
- `exporter.py`: Excel and text export.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Current Model Support

DCIM time-domain fitting:

- `simple`: 1RC
- `extended`: 2RC Extended Randles
- `warburg`: 2RC + `sigma_W * sqrt(t)`
- `joint_warburg`: ramp + CC joint fit with fitted Rs
- `relaxation`: CC Warburg + current-off relaxation, requires current interruption data
- `3rc`: 3RC time-domain fit

EIS fitting:

- `2RC`
- `2RC_CPE`
- `3RC`
- `Randles_W`

## Important Conventions

- Internal current unit: A.
- Internal resistance unit: ohm.
- Display resistance unit: mOhm.
- `render_tab_*()` functions are called without arguments.
- `render_manual_range()` returns `(p2_override, window_s, relax_window_s)`.
- Nyquist/EIS Rs source priority:
  1. fitted EIS Rs when available,
  2. capacitive arc minimum Re(Z),
  3. DCIM 2-wire Rs.

## Known Physical Limits

DCIM 2-wire Rs and EIS Kelvin Rs are not directly identical. A fixed offset can remain due to wiring/contact resistance and early-sample RC charging. Warburg and relaxation models reduce parameter mixing, but HPPC/current-off data is the most reliable path for separating RC relaxation from diffusion drift.
