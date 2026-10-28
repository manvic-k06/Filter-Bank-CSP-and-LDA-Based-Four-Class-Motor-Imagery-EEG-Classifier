`timescale 1ns/1ps
// Dataset-derived testbench for the 96-feature LDA RTL.
// Inputs/labels are generated from held-out dataset trials by export_dataset_testbench.py.
module tb_dataset_lda;
  parameter integer MAX_TRIALS = 1000;
  parameter integer DEFAULT_TRIALS = 82;   // all held-out trials of subject 7
  parameter integer N_FEATURES = 96;
  parameter FEATURE_FILE = "test_features.mem";
  parameter LABEL_FILE = "test_labels.mem";
  parameter EXPECTED_FILE = "expected_predictions.mem";
  parameter WEIGHT_FILE = "lda_weights.mem";
  parameter BIAS_FILE = "lda_bias.mem";
  reg clk=0, rst=1, feature_valid=0;
  reg signed [15:0] feature_in=0;
  wire class_valid; wire [1:0] predicted_class;
  wire signed [47:0] score0,score1,score2,score3;
  reg [15:0] feature_mem [0:MAX_TRIALS*N_FEATURES-1];
  reg [3:0] label_mem [0:MAX_TRIALS-1];
  reg [3:0] expected_mem [0:MAX_TRIALS-1];
  integer n_trials=0, i, t, w, got, correct=0, rtl_ref_match=0, mismatch=0;
  always #5 clk=~clk;
  csp_lda_core #(.WEIGHT_FILE(WEIGHT_FILE), .BIAS_FILE(BIAS_FILE)) dut(
    .clk(clk),.rst(rst),.feature_valid(feature_valid),.feature_in(feature_in),
    .class_valid(class_valid),.predicted_class(predicted_class),
    .score0(score0),.score1(score1),.score2(score2),.score3(score3));
  initial begin
    if (!$value$plusargs("NTRIALS=%d", n_trials)) n_trials=DEFAULT_TRIALS;
    if (n_trials<1 || n_trials>MAX_TRIALS) $fatal(1,"NTRIALS must be 1..%0d",MAX_TRIALS);
    $readmemh(FEATURE_FILE, feature_mem); $readmemh(LABEL_FILE, label_mem); $readmemh(EXPECTED_FILE, expected_mem);
    if (^feature_mem[0] === 1'bx || ^dut.weights[0] === 1'bx)
      $fatal(1,"Dataset .mem files were NOT loaded (check file paths / generics)");
    repeat(5) @(negedge clk); rst=0;
    for (t=0; t<n_trials; t=t+1) begin
      for (i=0; i<N_FEATURES; i=i+1) begin
        @(negedge clk); feature_in=feature_mem[t*N_FEATURES+i]; feature_valid=1;
      end
      @(negedge clk); feature_valid=0; feature_in=0;
      // class_valid is a 1-cycle pulse; poll for it for a bounded number of cycles.
      got=0;
      for (w=0; w<8 && !got; w=w+1) begin
        if (class_valid===1'b1) got=1; else @(negedge clk);
      end
      if (!got) $fatal(1,"No class_valid after trial %0d",t);
      #1;
      if (predicted_class === expected_mem[t]) rtl_ref_match=rtl_ref_match+1;
      else begin mismatch=mismatch+1; $display("RTL/reference mismatch trial %0d: RTL=%0d expected=%0d",t,predicted_class,expected_mem[t]); end
      if (predicted_class === label_mem[t]) correct=correct+1;
      $display("trial=%0d predicted=%0d dataset_label=%0d fixedpoint_reference=%0d",t,predicted_class,label_mem[t],expected_mem[t]);
      @(negedge clk);
    end
    $display("--------------------------------------------------");
    $display("Dataset trials tested: %0d",n_trials);
    $display("RTL vs fixed-point SW: %0d/%0d",rtl_ref_match,n_trials);
    $display("Accuracy vs dataset labels: %0d/%0d = %0.2f%%",correct,n_trials,100.0*correct/n_trials);
    $display("RTL/reference mismatches: %0d",mismatch);
    if (mismatch != 0) $fatal(1,"RTL does not match the fixed-point reference");
    $display("Labels are held-out dataset labels; testbench inputs are not synthetic.");
    $finish;
  end
  initial begin #200000000; $fatal(1,"Simulation timeout"); end
endmodule
