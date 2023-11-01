function [simul_hourly_flow, simul_hourly_spilling, simul_flow_dispatch, simul_flow_dispatch_correction, simul_revenue, balancing_fine, simul_water_level, simul_spilling] = ...
    use_Qtable_2018(Q_table, ...
    init_water_level_ind, stateWaterHeight, actionDispatch, day_ind, diff_ind, ...
    alpha,evap, ...
    inflow,optimRevenue_2018,optimFlow_2018,averageUpPricePremium)

%%%%%%% Use the Q-table for inflow year 2018 flow dispatch optimization %%%%%%%

seconds_in_hour = 3600;
hours_in_day = 24;

water_level_ind = init_water_level_ind; 
day_init = 1;
state = [day_ind(day_init),diff_ind(day_init),water_level_ind];
d = 365;
hours = d*hours_in_day;
% Simulated flow
simul_flow_dispatch = zeros(d,1); 
simul_flow_dispatch_correction = zeros(d,1); simul_hourly_flow = zeros(hours,1);
% Simulated spilling
simul_spilling =  zeros(d,1); simul_hourly_spilling =  zeros(hours,1);
% Simulated water level
simul_water_level = zeros(d,1); 
% Simulated reward
simul_revenue = zeros(d,1); 
% Check if there are states with no Q-values for actions
zero_cell =  zeros(d,1);
% Balancing fine if the water level would go below minimum
balancing_fine =  zeros(d,1);
% Value is the max_a Q
V_table = zeros(length(Q_table(:,1,1,1)),length(Q_table(1,:,1,1)),length(Q_table(1,1,:,1)));

for day = 1:d
    % Policy:
    if sum(Q_table(state(1),state(2),state(3),:)) == 0
        zero_cell(day) = 1;
        action_ind = randi([1 length(actionDispatch)],1,1);
    else
        [V_table(state(1),state(2),state(3)),action_ind] = max(Q_table(state(1),state(2),state(3),:));
    end
    % Average hourly flow (m^3/s)
    simul_flow_dispatch(day) = actionDispatch(action_ind); 

    % Total daily revenue (€)
    simul_revenue(day) = optimRevenue_2018(day_ind(day),action_ind);

    % Water level:
    simul_water_level(day) = stateWaterHeight(state(3));
    net_flow = (inflow(day) - simul_flow_dispatch(day))*seconds_in_hour*hours_in_day; % (m^3)
    delta_height = alpha*net_flow; % (cm)
    next_water_level = stateWaterHeight(state(3)) + delta_height - evap(day_ind(day)); % (cm)
    % Check if next day's water level would go above max: Spilling
    if next_water_level > stateWaterHeight(end) 
        spill = next_water_level - stateWaterHeight(end); % height over max in (cm)
        spill_m3 = (1/alpha)*spill; % spilling in (m3)
        spill_m3_per_s = spill_m3/(hours_in_day*seconds_in_hour); % spilling per hour-of-day in (m3/s)
        simul_hourly_spilling((1+(day-1)*hours_in_day):(hours_in_day+(day-1)*hours_in_day)) = ones(hours_in_day,1)*spill_m3_per_s;
        % Hourly flow (m^3/s) for each hour of day 
        simul_hourly_flow((1+(day-1)*hours_in_day):(hours_in_day+(day-1)*hours_in_day)) = optimFlow_2018(:,day,action_ind);
    % Check if water level between min and max
    elseif next_water_level >= stateWaterHeight(1)
        % Hourly flow (m^3/s) for each hour of day 
        simul_hourly_flow((1+(day-1)*hours_in_day):(hours_in_day+(day-1)*hours_in_day)) = optimFlow_2018(:,day,action_ind);
    % Check if next day's water level would go below min: 
    % (buy the missing energy at expected up-regulating price)
    else % next_water_level < stateWaterHeight(1)
        missing_water = (next_water_level - stateWaterHeight(1))*(1/alpha); % (m^3)
        new_daily_total = simul_flow_dispatch(day)*(seconds_in_hour*hours_in_day) + missing_water; % (m^3)
        new_average_dispatch = new_daily_total/(seconds_in_hour*hours_in_day); % (m^3/s)
        [~,corrected_action_ind] = min(abs(actionDispatch-new_average_dispatch));
        % Hourly flow (m^3/s) for each hour of day 
        simul_hourly_flow((1+(day-1)*hours_in_day):(hours_in_day+(day-1)*hours_in_day)) = optimFlow_2018(:,day,corrected_action_ind);
        simul_flow_dispatch_realized = actionDispatch(corrected_action_ind); % average hourly (m^3/s)
        simul_flow_dispatch_correction(day) = simul_flow_dispatch_realized-simul_flow_dispatch(day);
        real_revenue = optimRevenue_2018(day_ind(day),corrected_action_ind);
        balancing_fine(day) = (1+averageUpPricePremium)*(real_revenue - simul_revenue(day));
        next_water_level = stateWaterHeight(1);
    end
    % (cm) back to (m3/s): 
    cm_to_m3_per_second = 1/(alpha*seconds_in_hour*hours_in_day);
    simul_spilling(day) = min(0,(stateWaterHeight(end) -  next_water_level)*cm_to_m3_per_second);
    [~,next_water_level_ind] = min(abs(stateWaterHeight-next_water_level));
    % Next state:
    if day == d
        next_day = 1;
    else
        next_day = day + 1;
    end
    new_state = [day_ind(next_day),diff_ind(next_day),next_water_level_ind]; 
    state = new_state;
end
end