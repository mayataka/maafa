from sympy import symbols, sin, cos, diff, simplify

# parameters 
g, M, m, l = symbols(['g', 'M', 'm', 'l'])
# state and input
y, th, dy, dth, u = symbols(['y', 'th', 'dy', 'dth', 'u'])

sin_th = sin(th)
cos_th = cos(th)
cin = (u+m*l*dth**2*sin_th) / (M+m)
ddth = (g*sin_th-cos_th*cin) / (l*(4./3.-(m*cos_th**2)/(M+m)))
ddy = cin - (m*l*ddth*cos_th) / (M+m)

ddth_y = simplify(diff(ddth, y))
ddth_th = simplify(diff(ddth, th))
ddth_dy = simplify(diff(ddth, dy))
ddth_dth = simplify(diff(ddth, dth))
ddth_u = simplify(diff(ddth, u))
ddy_y = simplify(diff(ddy, y))
ddy_th = simplify(diff(ddy, th))
ddy_dy = simplify(diff(ddy, dy))
ddy_dth = simplify(diff(ddy, dth))
ddy_u = simplify(diff(ddy, u))

print("ddth_y:", ddth_y)
print("ddth_th:", ddth_th)
print("ddth_dy:", ddth_dy)
print("ddth_dth:", ddth_dth)
print("ddth_u:", ddth_u)
print("ddy_y:", ddy_y)
print("ddy_th:", ddy_th)
print("ddy_dy:", ddy_dy)
print("ddy_dth:", ddy_dth)
print("ddy_u:", ddy_u)