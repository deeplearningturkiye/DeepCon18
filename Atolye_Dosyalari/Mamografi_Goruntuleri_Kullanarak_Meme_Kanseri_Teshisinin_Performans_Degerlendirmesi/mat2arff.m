function [ success_value ] = mat2arff( datas, labels, arff_filename, relation, classes )
%   MAT2ARFF is a function to convert the .mat files containing extracted
%   features into the .arff file format for use with the WEKA3 datamining
%   toolkit. The .mat files have to contain a feature matrix eiter two or
%   three dimensional. möat
%EXAMPLE:uuR(SL329Dataset,SL329Label,'SL329ARFF.arff','a','0,1')
%   (c) Jens Bandener 2011 [Jens.Bandener@ruhr-uni-bochum.de]
%                                       script version v0.1 (02-dec-2011)
%   
%
%   .mat file conditions:
%   ------------------------------------------------------------------
%   requires following variables stored inside the .mat file.
%   [string]        AudioType
%   [double matrix] featureMatrix
%   AudioType containing the attribute for the features in this .mat file.
%   featureMatrix containing the feature data as a 2-dim or 3-dim double
%   precision matrix
%   All other variables stored in the .mat file will not be used in this
%   version.
%
%
%   Input arguments:
%   ------------------------------------------------------------------
%   mat_filename:       path and filename of .mat file
%   arff_filename:      path and filename of output file
%                       ! if file already exists features of the
%                       mat_filename are attached to the arff file. Does
%                       only work for mat files with the same number and
%                       order of features. Header informations of the arff
%                       file are not updated if file already exists
%   feature_var:        name of the matrix variable containing features
%                       The last index of feature_var is used as a
%                       separator for the features. (if otherwise use is
%                       required this script has to be adapted). 
%   relation:           string containing the arff relation
%
%   Ouput arguments:
%   -------------------------------------------------------------------
%   success_value:      returs value "1" if arff file has been written
%                       successfully to the filesystem. Otherwise it
%                       returns "0"

fid = fopen(arff_filename, 'a+');

feature_var = datas'; % TODO replace with valid code !! 


if isempty(fread(fid))
    disp('creating arff file...');
    disp('generating header information...');
    % save comments to the file
    %string_value = sprintf('%% %s  ','');
    fprintf(fid, '%% %s \n', 'created with mat2arff skript');
    fprintf(fid, '%% created on: %s \n\n', datestr(now));
    fprintf(fid, '@RELATION %s \n', relation);
    % create ATTRIBUTE entry for each variable
    if(length(size(feature_var)) == 2)
        for x=1:size(feature_var,1)
            string_value = sprintf('var%i',x);
            fprintf(fid, '@ATTRIBUTE %s NUMERIC\n', string_value);
        end
    elseif (length(size(feature_var)) == 3)
            for x=1:size(feature_var,1)
                for y=1:size(feature_var,2)
                    string_value = sprintf('var%i_%i',x,y);
                    fprintf(fid, '@ATTRIBUTE %s NUMERIC\n', string_value);
                end
            end
    else
        disp('feature_var not in the correct format...exiting');
        success_value = 0;
        close;
    end
    
    fprintf(fid, '@ATTRIBUTE class {%s} \n', classes);
    fprintf(fid, '@DATA \n');
    disp('writing feature data from mat-file...');
end
    % append dataset for each fature-set
    
    if(length(size(feature_var)) == 2)
        for z=1:size(feature_var,2) % iterate over data set
            for x=1:size(feature_var,1)
                fprintf(fid, '%d,', feature_var(x,z));
            end
            fprintf(fid, '%d\n', labels(z)); % new-line at the end of each dataset
        end
    elseif (length(size(feature_var)) == 3)
        for z=1:size(feature_var,3) % iterate over data set
            for x=1:size(feature_var,1)
                for y=1:size(feature_var,2)
                    fprintf(fid, '%d,', feature_var(x,y,z));
                end
            end
            fprintf(fid, '%d\n', labels(z)); % new-line at the end of each dataset
        end
    end

disp('data block added to arff file...');
fclose(fid);
end


%mat2arff('Temp.mat',filename ,'a','Disorder,Order');