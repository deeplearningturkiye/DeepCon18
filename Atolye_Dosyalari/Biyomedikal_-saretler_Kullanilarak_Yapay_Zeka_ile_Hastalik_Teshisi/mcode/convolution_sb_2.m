
function SC_Conv=convolution_sb_2(SC)
%Convolution 5x5
               h=[1 0 0 0 1;0 1 0 1 0;0 0 1 0 0;0 1 0 1 0;1 0 0 0 1 ] ;
                for i=1:size(SC,1)-4
                    for j=1:size(SC,2)-4
                        x=SC(i,j)*h(1,1)+SC(i,j+1)*h(1,2)+SC(i,j+2)*h(1,3)+SC(i,j+3)*h(1,4)+SC(i,j+4)*h(1,5);
                        y=SC(i+1,j)*h(2,1)+SC(i+1,j+1)*h(2,2)+SC(i+1,j+2)*h(2,3)+SC(i+1,j+3)*h(2,4)+SC(i+1,j+4)*h(2,5);
                        z=SC(i+2,j)*h(3,1)+SC(i+2,j+1)*h(3,2)+SC(i+2,j+2)*h(3,3)+SC(i+2,j+3)*h(3,4)+SC(i+2,j+4)*h(3,5);
                        w=SC(i+3,j)*h(4,1)+SC(i+3,j+1)*h(4,2)+SC(i+3,j+2)*h(4,3)+SC(i+3,j+3)*h(4,4)+SC(i+3,j+4)*h(4,5);
                        l=SC(i+4,j)*h(5,1)+SC(i+4,j+1)*h(5,2)+SC(i+4,j+2)*h(5,3)+SC(i+4,j+3)*h(5,4)+SC(i+4,j+4)*h(5,5);
                        SC_Conv(i,j)=x+y+z+w+l;
                    end
                end
end