s = 0.1;
Point(1) = {-1, 0, 0, s};
Point(2) = {0, 0, 0, s};
Point(3) = {0, -0.2, 0, s};
Point(4) = {2, -0.2, 0, s};
Point(5) = {2, 0, 0, s};
Point(6) = {2, 0.4, 0, s};
Point(7) = {0, 0.4, 0, s};
Point(8) = {-1, 0.4, 0, s};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {2, 5};
Line(10) = {2, 7};
Curve Loop(1) = {1, 10, 7, 8};
Plane Surface(1) = {1};
Curve Loop(2) = {2, 3, 4, -9};
Plane Surface(2) = {2};
Curve Loop(3) = {5, 6, -10, 9};
Plane Surface(3) = {3};


Transfinite Curve {8, 10, 5} = 60 Using Progression 1;
Transfinite Curve {2, 4} = 30 Using Progression 1;
Transfinite Curve {1, 7} = 75 Using Progression 1;
Transfinite Curve {3, 9, 6} = 150 Using Progression 1;


Transfinite Surface {1};
Transfinite Surface {3};
Transfinite Surface {2};
Recombine Surface {1, 3, 2};
Extrude {0, 0, 0.8} {
  Surface{1}; Surface{3}; Surface{2}; Layers {80}; Recombine;
}
Physical Surface("inlet", 77) = {31};
Physical Surface("outlet", 78) = {41, 71};
Physical Surface("walls_top", 79) = {27, 45};
Physical Surface("walls_bot", 80) = {19, 63, 67};
Physical Surface("sides", 81) = {32, 76, 54, 2, 3, 1};
Physical Volume("internal", 82) = {1, 2, 3};
