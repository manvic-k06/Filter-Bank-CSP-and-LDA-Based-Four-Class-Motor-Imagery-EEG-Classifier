`timescale 1ns/1ps
// Synthesizable serial-feature LDA core for the notebook's 96-feature model.
// Input feature: signed Q8.8. LDA weights: signed Q?.14. Biases: signed Q?.14.
// One feature_valid pulse per feature, exactly 96 features per trial.
module csp_lda_core #(
    parameter integer N_FEATURES = 96,
    parameter integer N_CLASSES  = 4,
    parameter integer FEATURE_W  = 16,
    parameter integer WEIGHT_W   = 16,
    parameter integer ACC_W      = 48,
    parameter integer FEAT_FRAC  = 8,
    parameter WEIGHT_FILE = "lda_weights.mem",
    parameter BIAS_FILE   = "lda_bias.mem"
)(
    input wire clk,
    input wire rst,
    input wire feature_valid,
    input wire signed [FEATURE_W-1:0] feature_in,
    output reg class_valid,
    output reg [1:0] predicted_class,
    output reg signed [ACC_W-1:0] score0,
    output reg signed [ACC_W-1:0] score1,
    output reg signed [ACC_W-1:0] score2,
    output reg signed [ACC_W-1:0] score3
);
    reg signed [WEIGHT_W-1:0] weights [0:N_FEATURES*N_CLASSES-1];
    reg signed [ACC_W-1:0] biases [0:N_CLASSES-1];
    reg signed [ACC_W-1:0] accum [0:N_CLASSES-1];
    reg [6:0] idx;
    reg active;
    integer k;

    wire signed [WEIGHT_W-1:0] w0 = weights[idx*N_CLASSES+0];
    wire signed [WEIGHT_W-1:0] w1 = weights[idx*N_CLASSES+1];
    wire signed [WEIGHT_W-1:0] w2 = weights[idx*N_CLASSES+2];
    wire signed [WEIGHT_W-1:0] w3 = weights[idx*N_CLASSES+3];
    wire signed [FEATURE_W+WEIGHT_W-1:0] p0 = feature_in * w0;
    wire signed [FEATURE_W+WEIGHT_W-1:0] p1 = feature_in * w1;
    wire signed [FEATURE_W+WEIGHT_W-1:0] p2 = feature_in * w2;
    wire signed [FEATURE_W+WEIGHT_W-1:0] p3 = feature_in * w3;
    // Product Q(8+14); shift by feature fraction (8) to leave scores in Q14.
    wire signed [ACC_W-1:0] pe0 = p0;   // signed assignment sign-extends
    wire signed [ACC_W-1:0] d0 = pe0 >>> FEAT_FRAC;
    wire signed [ACC_W-1:0] pe1 = p1;   // signed assignment sign-extends
    wire signed [ACC_W-1:0] d1 = pe1 >>> FEAT_FRAC;
    wire signed [ACC_W-1:0] pe2 = p2;   // signed assignment sign-extends
    wire signed [ACC_W-1:0] d2 = pe2 >>> FEAT_FRAC;
    wire signed [ACC_W-1:0] pe3 = p3;   // signed assignment sign-extends
    wire signed [ACC_W-1:0] d3 = pe3 >>> FEAT_FRAC;
    wire signed [ACC_W-1:0] base0 = active ? accum[0] : biases[0];
    wire signed [ACC_W-1:0] base1 = active ? accum[1] : biases[1];
    wire signed [ACC_W-1:0] base2 = active ? accum[2] : biases[2];
    wire signed [ACC_W-1:0] base3 = active ? accum[3] : biases[3];
    wire signed [ACC_W-1:0] next0 = base0 + d0;
    wire signed [ACC_W-1:0] next1 = base1 + d1;
    wire signed [ACC_W-1:0] next2 = base2 + d2;
    wire signed [ACC_W-1:0] next3 = base3 + d3;

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE, biases);
    end

    always @(posedge clk) begin
        if (rst) begin
            idx <= 0; active <= 0; class_valid <= 0; predicted_class <= 0;
            score0 <= 0; score1 <= 0; score2 <= 0; score3 <= 0;
            for (k=0; k<N_CLASSES; k=k+1) accum[k] <= 0;
        end else begin
            class_valid <= 0;
            if (feature_valid) begin
                accum[0] <= next0; accum[1] <= next1;
                accum[2] <= next2; accum[3] <= next3;
                if (!active) begin
                    active <= 1;
                    idx <= 1;
                end else if (idx == N_FEATURES-1) begin
                    score0 <= next0; score1 <= next1; score2 <= next2; score3 <= next3;
                    class_valid <= 1; active <= 0; idx <= 0;
                    if ((next1 > next0) && (next1 >= next2) && (next1 >= next3)) predicted_class <= 1;
                    else if ((next2 > next0) && (next2 > next1) && (next2 >= next3)) predicted_class <= 2;
                    else if ((next3 > next0) && (next3 > next1) && (next3 > next2)) predicted_class <= 3;
                    else predicted_class <= 0;
                end else idx <= idx + 1'b1;
            end
        end
    end
endmodule
