#ifndef DEVITO_MATH_H
#define DEVITO_MATH_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double coeffs[401];

double bessi1(double x){

   double z, numerator, denominator;
   z = x * x;
   numerator = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*(z* 0.210580722890567e-22  + 0.380715242345326e-19 ) + 0.479440257548300e-16) + 0.435125971262668e-13 ) + 0.300931127112960e-10) + 0.160224679395361e-7  ) + 0.654858370096785e-5)  + 0.202591084143397e-2  ) + 0.463076284721000e0)   + 0.754337328948189e2   ) + 0.830792541809429e4)   + 0.571661130563785e6   ) + 0.216415572361227e8)   + 0.356644482244025e9   ) + 0.144048298227235e10);

   denominator = (z*(z*(z-0.307646912682801e4)+0.347626332405882e7)-0.144048298227235e10);

        return -numerator/denominator;

}

double kaiser(double t){

    double r, beta, param1, a, b;

    r = 9.0;
    beta = 6.31;
    param1 = beta * sqrt(1 - ((4*pow(t,2))/pow((r-1),2)));
    a = bessi1(param1);
    b = bessi1(beta);

    return a/b;

}

int populate(){
    double i = 0;
    float pi = 3.14159265358979F;
    int ind = 0;
    while(i<=4){
        if(i == 0)
             coeffs[ind] = 1;
        else 
             coeffs[ind] = kaiser(i)*(sin(pi*i)/(pi*i));
        
       i = i + 0.01;
       ind++;
    }
    return 0;
}

double acessCoeffs(double dist){
    double step = 0.01;
    double rest;
    
    rest = fmod(dist*100, step*100);

    if(rest == 0)
        return coeffs[abs((int)(dist/step))];
    if(rest < 0.5){ 
        return coeffs[abs((int)(dist/step))];
    }
    return coeffs[abs((int)((dist/step)+1))];

}
#endif

