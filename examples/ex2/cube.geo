
s = 0.1;

Point(1) = {0, 0, 0, s};
Point(2) = {1, 0, 0, s};
Point(3) = {1, 1, 0, s};
Point(4) = {0, 1, 0, s};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};



Transfinite Curve{1, 2, 3, 4} = 11;
Transfinite Surface{1};

Recombine Surface{1};

//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {10}; Recombine;
}

Physical Surface("top") = {21};
Physical Surface("bottom") = {13, 1, 26};
Physical Surface("left") = {25};
Physical Surface("right") = {17};
Physical Volume("internal") = {1}; 