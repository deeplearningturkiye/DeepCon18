function pooling=pooling_sb(SC)
 %pooling 2x2
                    m=1;
                     for i=1:2:size(SC,1)-1
                         n=1;
                        for j=1:2:size(SC,2)-1
                           pooling(m,n)=max(max([SC(i,j) SC(i,j+1);SC(i+1,j) SC(i+1,j+1)]));
                            n=n+1;
                        end
                        m=m+1;
                    end
end