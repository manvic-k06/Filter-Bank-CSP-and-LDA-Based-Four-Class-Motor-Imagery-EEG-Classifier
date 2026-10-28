# Filter-Bank-CSP-and-LDA-Based-Four-Class-Motor-Imagery-EEG-Classifier
# Motor-Imagery EEG Classifier on FPGA

4-class motor-imagery EEG classifier (BCI Competition IV 2a). Python (Colab) trains the model
and makes the features; the LDA classifier is written in Verilog and simulated in Vivado
(part xc7a200tfbv676-1).

## Files

```
constraints/clk.xdc
dataset/dataset_testbench_info.json
dataset/expected_predictions.mem
dataset/lda_bias.mem
dataset/lda_weights.mem
dataset/test_features.mem
dataset/test_labels.mem
python_script_Algo/bci_csp_lda_fb.ipynb
python_script_Algo/bci_csp_lda_fb.py
python_script_Algo/export_dataset_testbench.py
python_script_Algo/make_dataset_colab.ipynb
rtl/csp_lda_core.v
tb/tb_dataset_lda.v
```

## 1. Train the model (Colab)

1. Open `python_script_Algo/bci_csp_lda_fb.ipynb` in Google Colab.
2. Run all cells from top to bottom. The model of each subject is saved as
   `results_fb/model_s01.npz` ... `model_s09.npz`.

## 2. Get the weights and testbench data

In a new Colab cell (upload `python_script_Algo/export_dataset_testbench.py` to Colab first):

```
!python export_dataset_testbench.py results_fb/model_s07.npz --out dataset_new --max-trials 0
```

This creates the five files used in `dataset/`: `lda_weights.mem`, `lda_bias.mem`,
`test_features.mem`, `test_labels.mem`, `expected_predictions.mem`
(plus `dataset_testbench_info.json`). Download `dataset_new` and copy the files into `dataset/`.
Change `s07` for another subject. The `dataset/` folder already contains subject 7.

If the number of trials changes, set `DEFAULT_TRIALS` in `tb/tb_dataset_lda.v` to the `trials`
value in `dataset_testbench_info.json`.

## 3. Vivado

Use a short project path without spaces, for example `C:\bci_fpga`.

1. Create an RTL project and select the part `xc7a200tfbv676-1`.
2. Add Sources > Add design sources: `rtl/csp_lda_core.v`.
3. Add Sources > Add constraints: `constraints/clk.xdc`.
4. Add Sources > Add simulation sources: `tb/tb_dataset_lda.v` and the five `.mem` files from `dataset/`.
5. Run Simulation > Run Behavioral Simulation. The console must show
   `RTL/reference mismatches: 0`.
6. Run Synthesis, then Run Implementation, then open the Timing, Utilization and Power reports.

If simulation says the `.mem` files were not loaded, copy them to
`<project>.sim/sim_1/behav/xsim/`.

## Dataset

Not included. BCI Competition IV 2a: https://www.bbci.de/competition/iv/
