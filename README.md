# -IMPULSE-AND-STEP-RESPONSE-ANALYSIS-OF-DISCRETE-TIME-SYSTEMS-USING-PYTHON
1. PROJECT PURPOSE:
 ANALYZE IMPULSE AND STEP RESPONSES
 To examine the response behavior of Linear
 Time Invariant (LTI) systems
 To observe the responses to input signals
 (δ[n], u[n])
 To make a comparative analysis of FIR and IIR
 systems
Lcera Tech
 2. Input Sgnals and Systems Used
 Input Signals:
 Unit Impulse (δ[n]): 1 only for n=0
 Unit Step (u[n]): 1 for n ≥ 0
 Systems:
 FIR (Moving Average): h1 = [1/3, 1/3, 1/3]
 IIR (Decreasing Exponential): h2[n] = (0.8)^n,
 n = 0…19
3. CONVOLUTİON APPLİCATİON AND
 CALCULATİON METHOD
 Manual convolution with
 custom_convolve(x, h) function
 Comparison with numpy.convolve to
 check accuracy
 Outputs calculated for both impulse and
 step signals
4. VİSUAL RESULTS AND TECHNİCAL COMMENTS
 FIR System:
 Impulse Response: Constant amplitude lasting 3 samples
 Step Response: Linear increase, reaching a constant level
 IIR System:
 Impulse Response: Exponentially decaying signal
 Step Response: Steadily increasing, reaching saturation output
 Visualizations are presented as stem plots with matplotlib.pyplot.
