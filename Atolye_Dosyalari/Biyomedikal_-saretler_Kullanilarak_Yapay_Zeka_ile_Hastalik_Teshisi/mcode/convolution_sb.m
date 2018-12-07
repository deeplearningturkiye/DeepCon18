
function SC_Conv=convolution_sb(SC)
%Convolution 3x3
               h=[1  0 -1;1 0 -1;1 0 -1] ;
                 for i=1:size(SC,1)-2
                    for j=1:size(SC,2)-2
                        x=SC(i,j)*h(1,1)+SC(i,j+1)*h(1,2)+SC(i,j+2)*h(1,3);
                        y=SC(i+1,j)*h(2,1)+SC(i+1,j+1)*h(2,2)+SC(i+1,j+2)*h(2,3);
                        z=SC(i+2,j)*h(3,1)+SC(i+2,j+1)*h(3,2)+SC(i+2,j+2)*h(3,3);
                        SC_Conv(i,j)=x+y+z;
                    end
                end
end