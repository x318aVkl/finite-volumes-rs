Point(1) = {-1, -0.6, 0, 1.0};
Point(2) = {-0, -0.6, 0, 1.0};
Point(3) = {0, -0.3, 0, 1.0};
Point(4) = {0.1, -0.3, 0, 1.0};
Point(5) = {0.1, -1.0, 0, 1.0};
Point(6) = {3, -1.0, 0, 1.0};
Point(7) = {3, 1.0, 0, 1.0};
Point(8) = {-1, 1.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Curve Loop(1) = {8, 1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};
//+
Physical Curve("inlet", 9) = {8};
//+
Physical Curve("outlet", 10) = {6};
//+
Physical Curve("wall", 11) = {1, 2, 3, 4, 5};
//+
Physical Curve("sides", 12) = {7};
//+
Physical Surface("internal", 13) = {1};
