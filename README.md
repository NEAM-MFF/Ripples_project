# Class-specific coupling of single-unit firing to cortical ripples in macaque V1

The analysis of ripple band oscillations and their generation mechanisms.
The scripts for data preprocessing can be run and paralelised via .sh submit files.
Main function file: functions_analysis.py
Parameters: params_analysis.yaml
Preprocessing files necessary to recreate figures (+ associate Supp. Figures), in order:

Figure 1 + Figure 2 (Ripples):
- delta_rb_env_dict_figure1.py
- detect_ripples_df_one_arr_RS.py
- hypnogram_df.py
- ripple_trigg_df_create_new_RS.py
- ripple_trigg_spectra_create_RS.py

Figure 3 + Figure 4 (Single units):
- SUA_RS_prop_pkl_create.py
- shuffle_phases_df.py
- ripplw_trigg_phase_align_df_create_RS.py
- graph_SUA_preprocess.py

Figure 5:
- all the scripts labeled as _NATIM

The plots are created within Jupyter notebooks, labeled with numbers of Figures (F) and Supplmentary Figures (SF).

