Point(1) = {-1, -1, 0, 1.0};
Point(2) = {1, -1, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {-1, 1, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
Physical Curve("wall", 5) = {1, 2, 3, 4};
Physical Surface("internal", 6) = {1};

Transfinite Curve {4, 1, 2, 3} = 3 Using Progression 1;
Transfinite Surface {1};
Recombine Surface {1};
