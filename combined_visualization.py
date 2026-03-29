import importlib
import os
import sys

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Combined Visualizations", layout="wide")

    # Tell imported visualization modules they are being embedded.
    os.environ["STREAMLIT_EMBEDDED_MODE"] = "1"
    os.environ["FIRST_VIZ_HIDE_TITLE"] = "1"

    st.title("Combined Visualizations")

    top_panel, bottom_panel = st.container(border=True, height=680), st.container(border=True, height=680)

    with top_panel:
        st.markdown("##### First Visualization")
        # first_visualization renders at import-time, so execute it once per rerun.
        if "first_visualization" in sys.modules:
            importlib.reload(sys.modules["first_visualization"])
        else:
            importlib.import_module("first_visualization")

    with bottom_panel:
        st.markdown("##### Binary Visualization")
        import binary_visualization

        binary_visualization.main(show_title=False, compact=True)


if __name__ == "__main__":
    main()
