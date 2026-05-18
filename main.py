# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4.5",
#     "plotly>=6.7.0",
#     "streamlit>=1.57.0",
# ]
# ///

import streamlit.web.cli as stcli
import sys

sys.argv = ["streamlit", "run", "hrbf_tester.py"]
sys.exit(stcli.main())



