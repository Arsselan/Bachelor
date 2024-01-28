 
 
 
 
plot "results/timedomain_impact_plastic/Spline_n24_p2_CON_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
     "results/timedomain_impact_plastic_strong/Spline_n24_p2_CON_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
     "results/timedomain_impact_plastic/Spline_n24_p2_RS_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
     "results/timedomain_impact_plastic/Lagrange_n12_p2_CON_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
     "results/timedomain_impact_plastic/Lagrange_n12_p2_HRZ_dt5.000000e-06_final_disp.dat" u 1:2 w l

#plot "results/timedomain_impact_plastic/Spline_n48_p2_CON_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
#     "results/timedomain_impact_plastic_strong/Spline_n48_p2_CON_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
#     "results/timedomain_impact_plastic/Spline_n48_p2_RS_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
#     "results/timedomain_impact_plastic/Lagrange_n24_p2_CON_dt5.000000e-06_final_disp.dat" u 1:2 w l, \
#     "results/timedomain_impact_plastic/Lagrange_n24_p2_HRZ_dt5.000000e-06_final_disp.dat" u 1:2 w l

     
pause(-1)
