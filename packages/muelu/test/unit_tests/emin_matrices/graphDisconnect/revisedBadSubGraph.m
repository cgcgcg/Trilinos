%%  Edges marked with dashes are strong. Edges marked with dots are weak.
%%  Edges marked with plus signs were originall supposed to be weak edges
%%  (based on my guess of how MueLu picks root nodes). However, my guess
%%  was wrong, and so these ones are now set as strong. The main idea is
%%  that we want four perfect 2x2 aggregates to be chosen in each corner. 
%%  Getting this depends a bit on how MueLu chooses root nodes. The main
%%  idea is that SW aggregate has a strong connection to the NE aggregate.
%%  But since there are no fine edges connecting these two aggregagtes,
%%  a D0coarse edge will not be created between them. The same is also
%%  true for the interaction between the NW and SE aggregates. Thus,
%%  D0coarse consists of 4 edges that for the box that connects the 
%%  four aggregates.  However, the Pn basis function associated with 
%%  the SW aggregate has support within the NE aggregate (likewise for 
%%  the relationship between the NW and SE basis functions). For example,
%%  Dfine(4,:)*Pn is nonzero for coarse nodes 1 and 4 (assuming they
%%  are numbered lexicographically). Here, the 4th edge
%%  is the horizontal edge that connects node 5 with 6. Again, this is 
%%  due to the strong connection between the two associated aggregates.
%%  But since there is no strong connection to aggregate 2 and 3, 
%%  these do not appear in Dfine(4,:)*Pn . As there is no coarse edge
%%  between nodes 1 and 4, we can not solve the commuting relationship
%%
%%  Notice if there had been a strong connection between from either
%%  fine node 5 or 6 to a fine node in aggregate 3, then Dfine(4,:)*Pn 
%%  would have also included
%%  coarse node 3. In this scenario, it would be possible to solve
%%  this computing equation as the Dcoarsse path (1 --> 2) and
%%  (2-->3) would in fact lead to a connected sub-graph that includes
%%  coarse nodes 1, 3, and 4. 

%%                  G --------- G --------- G --------- H --------- H --------- H --------- I --------- I --------- I 
%%                22.          7.         23.         24.          8.         25.         / .26        9.           |27
%%                  .           .           .           .           .           .       /   .           .           |
%%                  .           .           .           .           .           .     /     .           .           |
%%                  .           .           .           .           .           .   /       .           .           |
%%                  .           .           .           .           .           . /         .           .           |
%%                  A --------- D --------- D --------- D --------- E --------- E  . . . . .F --------- F --------- C 
%%                16. \       17.          4.         18.          5.         19. \        6.         20.         / |21
%%                  .   \       .           .           .           .           .   \       .           .       /   |
%%                  .     \     .           .           .           .           .     \     .           .     /     |
%%                  .       \   .           .           .           .           .       \   .           .   /       |
%%                  .         \ .           .           .           .           .         \ .           . /         |
%%                  A --------- A --------- A --------- B --------- B --------- B --------- C --------- C --------- C 
%%                  10          1          11          12           2          13          14           3          15

An = sparse(27,27);
map = [10 1 11 12 2 13 14 3 15 ; 16 17 4 18 5 19 6 20 21 ; 22 7 23 24 8 25 26 9 27]';
e = 1.e-4;

% set all vertical connections to be weak and all horizontal connections to be strong
% later we add extra edges to match pattern and make the (19,6) edge weak.
for j=1:3, for i=1:9,
   if i ~= 9, An( map(i  ,j  ) , map(i+1,j  ) )= -1; An( map(i+1,j  ) , map(i  ,j  ) ) = -1; end;
   if i ~= 1, An( map(i  ,j  ) , map(i-1,j  ) )= -1; An( map(i-1,j  ) , map(i  ,j  ) ) = -1; end;
   if j ~= 1, An( map(i  ,j  ) , map(i  ,j-1) )= -e; An( map(i  ,j-1) , map(i  ,j  ) ) = -e; end;
   if j ~= 3, An( map(i  ,j  ) , map(i  ,j+1) )= -e; An( map(i  ,j+1) , map(i  ,j  ) ) = -e; end;
end; end;
An( 19,14) = -1; An( 14,19) = -1; 
An( 19,26) = -1; An( 26,19) = -1; 
An(  3,21) = -1; An( 21, 3) = -1; 
An( 16, 1) = -1; An(  1,16) = -1; 

An( 19, 6) = -e; An(6,19)   = -e;

An = An -  spdiags( An*ones(27,1), 0, 27,27);


nedges = 8*3+9*2;
Dfine = sparse(nedges,27);
c = 0;
for j=1:3, for i=1:8,
   c = c+1;   Dfine(c, map(i  ,j  )) = -1;  Dfine(c, map(i+1,j  )) =  1;
end; end;
for j=1:2, for i=1:9,
   c = c+1;   Dfine(c, map(i  ,j  )) = -1;  Dfine(c, map(i  ,j+1)) =  1;
end; end;

Ae = spalloc(nedges,nedges,7*nedges);
for i=1:8, for j= 1:2,  
   hor = (j-1)*8 + i; ver = 24+(j-1)*9+i;
   Ae(hor  ,hor  ) = Ae(hor  ,hor  ) + 1;
   Ae(hor+8,hor+8) = Ae(hor+8,hor+8) + 1;
   Ae(ver  ,ver  ) = Ae(ver  ,ver  ) + 1;
   Ae(ver+1,ver+1) = Ae(ver+1,ver+1) + 1;
   Ae(hor  ,hor+8) = Ae(hor  ,hor+8) - 1;
   Ae(hor+8,hor  ) = Ae(hor+8,hor  ) - 1;
   Ae(ver  ,ver+1) = Ae(ver  ,ver+1) - 1;
   Ae(ver+1,ver  ) = Ae(ver+1,ver  ) - 1;

   Ae(hor  ,ver  ) = Ae(hor  ,ver  ) - 1;       Ae(ver  ,hor  ) = Ae(ver  ,hor  ) - 1;
   Ae(hor  ,ver+1) = Ae(hor  ,ver+1) -    (-1); Ae(ver+1,hor  ) = Ae(ver+1,hor  ) -    (-1);
   Ae(hor+8,ver  ) = Ae(hor+8,ver  ) -    (-1); Ae(ver  ,hor+8) = Ae(ver  ,hor+8) -    (-1); 
   Ae(hor+8,ver+1) = Ae(hor+8,ver+1) - 1 ;      Ae(ver+1,hor+8) = Ae(ver+1,hor+8) -1 ; 
end;end;

mmwrite('A.dat',Ae);
mmwrite('An.dat',An);
mmwrite('D0h.dat',Dfine);
