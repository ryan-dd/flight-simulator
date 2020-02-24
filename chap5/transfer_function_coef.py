import parameters.aerosonde_parameters as MAV

Va = MAV.Va0
aphi1 = -1/2*MAV.rho*Va**2 * MAV.S_wing*MAV.b*MAV.C_p_p * MAV.b/(2*Va)
aphi2 = 1/2*MAV.rho*Va**2 * MAV.S_wing*MAV.b*MAV.C_p_delta_a

abeta1 = -MAV.rho*Va*MAV.S_wing * MAV.C_Y_beta / (2*MAV.mass)
abeta2 = MAV.rho*Va*MAV.S_wing * MAV.C_Y_delta_r / (2*MAV.mass)

atheta1 = -MAV.rho*Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_q * MAV.c/(2*Va)
atheta2 = -MAV.rho*Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_alpha
atheta3 = MAV.rho*Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_delta_e

aV1 = MAV.rho*Va_star*MAV.S_wing/MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e * delta_e_star) + MAV.rho * MAV.S_prop/MAV.mass * MAV.C_prop*Va_star
aV1 = aV1.item(0)
aV2 = MAV.rho * MAV.S_prop/MAV.mass * MAV.C_prop*MAV.k_motor**2*delta_t_star
aV3 = MAV.gravity