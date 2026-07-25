
xc = 0.35355339059;

Point(1) = {-2, -1, 0, 1.0};
Point(2) = {-2, 1, 0, 1.0};
Point(3) = {-1, -1, 0, 1.0};
Point(4) = {-1, 1, 0, 1.0};
Point(5) = {1, -1, 0, 1.0};
Point(6) = {1, 1, 0, 1.0};
Point(7) = {3, -1, 0, 1.0};
Point(8) = {3, 1, 0, 1.0};
Point(9) = {-0, -0, 0, 1.0};
Point(10) = {-xc, xc, 0, 1.0};
Point(11) = {xc, xc, 0, 1.0};
Point(12) = {xc, -xc, 0, 1.0};
Point(13) = {-xc, -xc, 0, 1.0};
Line(1) = {1, 3};
Line(2) = {3, 5};
Line(3) = {5, 7};
Line(4) = {7, 8};
Line(5) = {8, 6};
Line(6) = {6, 4};
Line(7) = {4, 2};
Line(8) = {2, 1};
Line(9) = {3, 4};
Line(10) = {5, 6};
Line(11) = {6, 11};
Line(12) = {5, 12};
Line(13) = {3, 13};
Line(14) = {4, 10};
Circle(15) = {12, 9, 11};
Circle(16) = {11, 9, 10};
Circle(17) = {10, 9, 13};
Circle(18) = {13, 9, 12};
Curve Loop(1) = {8, 1, 9, 7};
Plane Surface(1) = {1};
Curve Loop(2) = {10, -5, -4, -3};
Plane Surface(2) = {2};
Curve Loop(3) = {9, 14, 17, -13};
Plane Surface(3) = {3};
Curve Loop(4) = {13, 18, -12, -2};
Plane Surface(4) = {4};
Curve Loop(5) = {12, 15, -11, -10};
Plane Surface(5) = {5};
Curve Loop(6) = {6, 14, -16, -11};
Plane Surface(6) = {6};
Transfinite Curve {8, 9, 10, 4, 15, 17, 16, 18, 6, 2} = 60 Using Progression 1;
Transfinite Curve {14, 11, 12, 13} = 30 Using Progression 1;
Transfinite Curve {-1, 7} = 20 Using Progression 1.035;
Transfinite Curve {3, -5} = 60 Using Progression 1.02;
Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {3};
Recombine Surface {1, 3, 4, 5, 6, 2};
//+
Extrude {0, 0, 1} {
  Surface{1}; Surface{3}; Surface{5}; Surface{2}; Surface{4}; Surface{6}; Layers {60}; Recombine;
}
//+
Translate {0, 0, -0.5} {
  Volume{1}; Volume{2}; Volume{3}; Volume{4}; Volume{5}; Volume{6}; 
}
//+
Physical Surface("inlet", 151) = {27};
//+
Physical Surface("outlet", 152) = {101};
//+
Physical Surface("wall", 153) = {119, 75, 145, 57};
//+
Physical Surface("sides", 154) = {31, 127, 105, 97, 137, 39, 2, 106, 5, 84, 6, 150, 3, 62, 1, 40, 4, 128};
//+
Physical Volume("internal", 155) = {1, 2, 3, 4, 6, 5};
