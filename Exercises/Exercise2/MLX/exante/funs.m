classdef funs
methods(Static)
    
    function [] = layout()
        
        % set layout parameters
        set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
        set(groot, 'defaultLegendInterpreter','latex');
        set(groot, 'defaultTextInterpreter','latex');
        set(groot, 'defaultAxesFontSize', 12); 
        
    end    
    function [x, w] = GaussHermite(n)

        i   = 1:n-1;
        a   = sqrt(i/2);
        CM  = diag(a,1) + diag(a,-1);
        [V, L]   = eig(CM);
        [x, ind] = sort(diag(L));
        V       = V(:,ind)';
        w       = sqrt(pi) * V(:,1).^2;

    end
    function [x, w] = GaussHermite_lognorm(sigma,n)
        
        [x,w] = funs.GaussHermite(n);
                
        x = exp(x*sqrt(2)*sigma-0.5*sigma^2);
        w = w./sqrt(pi);
        
        % assert a mean of one
        assert(1-sum(w.*x) < 1e-8)
        
    end
    function x = nonlinspace(lo,hi,n,phi)
        % recursively constructs an unequally spaced grid.
        % phi > 1 -> more mass at the lower end of the grid.
        % lo can be a vector (x then becomes a matrix).

        x      = NaN(n,length(lo));
        x(1,:) = lo;
        for i = 2:n
            x(i,:) = x(i-1,:) + (hi-x(i-1,:))./((n-i+1)^phi);
        end

    end
    function [] = printfig(figin)

        fig = figure(figin);
        fig.PaperUnits = 'centimeters';   
        fig.PaperPositionMode = 'manual';
        fig.PaperPosition = [0 0 16 12];
        fig.PaperSize = [16 12];

        filename = ['figs\' get(fig,'name') ''];            
        print('-dpdf',['' filename '.pdf']);

    end
    function [log_sum, prob] = logsum(v1,v2,sigma)
        % calculates the log-sum and choice-probabilities.

        % 1. setup
        V           = [v1,v2];
        DIM         = size(v1,1);

        % 2. maximum over the discrete choices
        [mxm,id]    = max(V,[],2);

        % 3. logsum and probabilities
        if abs(sigma) > 1.0e-10

            % a. numerically robust log-sum
            log_sum  = mxm + sigma*log(sum(exp((V-mxm*ones(1,2))./sigma),2));

            % b. numerically robust probability
            prob = exp((V-log_sum*ones(1,2))./sigma);
            %Prob = exp(V/par.sigma)./repmat(sum(exp(V/par.sigma),2),1,2);

        else % no smoothing -> max-operator

            log_sum  = mxm;

            prob    = zeros(DIM,2);
            I       = cumsum(ones(DIM,1)) + (id-1)*DIM; % calculate linear index
            prob(I) = 1;

        end

    end

end
end

