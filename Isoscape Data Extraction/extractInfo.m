% Read in coords of all the sample points
coordPoints = readmatrix("latlon_points.csv");

% Read in isoP data
load("100km_1988_2010_noBounds.mat", "PI_stack");

data = [0, 0, 0, 0, 0];

% Loop through coordPoints and pull out the coords, also the index
for i=1:length(coordPoints)
   lat = coordPoints(i,1);
   lon = coordPoints(i,2);

   % Gather the cell length
   cellLength = size(PI_stack{1,i}, 1);
    
   % Create a repeating matrix of the coordinates
   coords = repmat([lat, lon], cellLength, 1);

   % Combine the coords and data to be appended from isoP
   isoP = horzcat(coords, PI_stack{1,i}(:, 1:3));
   
   % Add that to the data array
   data = vertcat(data,isoP);

   % Print Progress
   percentage = i*100/length(coordPoints);
   fprintf('The amount done is: %.2f%%\n', percentage);
end


writematrix(data, "isoP_SampleData.csv");