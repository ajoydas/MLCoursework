clc;
clearvars;
tic;

loc = dlmread('loc.txt');
[N, D] = size(loc);

clust = 2;

mu = cell(1);
sigma = cell(1);
g = cell(1);

for s=1:clust
    mu{s} = mean(loc);
    sigma{s} = cov(loc);
end

theta = rand(1, clust);
theta = theta / sum(theta);

for k=1:clust
    g{k} = theta(k) * mvnpdf(loc, mu{k}, sigma{k});
end

pi = cell2mat(g);
sumpi = sum(pi,2);
preLgl = log(sumpi);

for i=1:10000
     
    pij = pi ./ sumpi;
    
    for k=1:clust
        
        pi = pij(:,k);
        sumpi = sum(pi);
        
        mu{k} = (pi' * loc) / sumpi;
        
        sigma{k} = (pi' .* (loc - mu{k})' * (loc - mu{k})) / sumpi;
        
        theta(k) = sumpi / N;
        
    end
    
    for k=1:clust
        g{k} = theta(k) * mvnpdf(loc, mu{k}, sigma{k});
    end
    
    pi = cell2mat(g);
    sumpi = sum(pi,2);
    lgl = log(sumpi);
    
    disp(i);
    %disp([preLgl lgl]);
    
    if (preLgl == lgl)
        break;
    end
    
    preLgl = lgl;
    

end

dis = cell(1);

for k=1:clust
    dis{k} = mvnpdf(loc, mu{k}, sigma{k});
end

celldisp(dis);

scatter(loc(:,1), loc(:,2));

toc;