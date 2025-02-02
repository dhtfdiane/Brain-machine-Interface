%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the parameters for the position estimator based on the provided
% training data
%
% Input : training data
% Output : modelParameters : struct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [modelParameters] = positionEstimatorTraining(training_data)

    nb_neuron = size(training_data(1,1).spikes,1);
    dt = 20; % Bin size
    modelParameters=struct('param',struct(),'bin_size',dt,'nb_neuron',nb_neuron);
    param=repmat(struct(),1,8);

    %% Compute min size of recording through trials for one angle
    size_min=1000*ones(1,8);
    for j=1:8
        for k=1:size(training_data,1)
            if size(training_data(k,j).spikes,2)<size_min(j)
                size_min(j)=size(training_data(k,j).spikes,2); 
            end
        end
        param(j).size_min=size_min(j);
    end
    %% Compute ParameterModel
    % .dynamics : compute mean position and velocity on bin size dt for angle i
    for i=1:8
        mean_pos_vel=zeros(6,size_min(i));
        mean_pos_vel_bin=zeros(6,floor(size_min(i)/dt));

        for j=1:3  
            
            %jeme-hand-position (x,y,z) of each trial for angle i
            pos = zeros(size(training_data,1),size_min(i));        
            for k=1:size(training_data,1)
                t=training_data(k,i).handPos(j,:);
                pos(k,:)=t(1:size_min(i));
            end
            
            % mean_position and mean_velocity over all trials for j-eme position and angle i  
            mean_pos_vel(2*j-1,:)=mean(pos);
            mean_pos_vel(2*j,1:end-1)=0.5*(mean_pos_vel(2*j-1,2:end)-mean_pos_vel(2*j-1,1:end-1));   

            for iter=1:floor(size_min(i)/dt)
                
                i_start=(iter-1)*dt+1;
                i_end=min(iter*dt,size_min(i));

                %Mean position over iterieme-bin size dt
                mean_pos_vel_bin(2*j-1,iter) = mean(mean_pos_vel(2*j-1,i_start : i_end));

                %Mean velocity over iterieme-bin size dt
                mean_pos_vel_bin(2*j,iter) = mean(mean_pos_vel(2*j,i_start : i_end));
            end    
        end
        param(i).dynamics = mean_pos_vel_bin;
    end  
    
  % .firing_rates : compute mean firing rate over all trial on bin size dt for neuron i and angle j
    for j=1:8
        for i=1:nb_neuron

            %Spikes for neuron i angle j (each trial)
            spikes = zeros(size(training_data,1),size_min(j));
            for k=1:size(training_data,1)
                t=training_data(k,j).spikes(i,:);
                spikes(k,:)=t(1:size_min(j));
            end

            %mean of spikes for neuron i angle j (over all trials)
            means=mean(spikes);

            %firing rate for neuron i angle j on bin size dt
            firing_rate=zeros(1,floor(size_min(j)/dt));
            for iter=1:floor(size_min(j)/dt)
                firing_rate(iter) = mean(means((iter-1)*dt+1 : min(iter*dt,size_min(j))));
            end
            param(j).firing_rates(i,:) = firing_rate;
        end 
    end  
    modelParameters.param = param;
end
    
    
    