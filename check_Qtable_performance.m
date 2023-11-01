function [zeroCells,waterLevelBreaks,totalReward] = check_Qtable_performance(Q_table, ...
    init_water_level_ind, stateWaterHeight, actionDispatch, ...
    day_ind, diff_ind, alpha, evap, discount_factor, ...
    inflow,optimRevenue_2018,training_period_days)

%%%%%%% Use the Q-table for 2000 - 2017 flow dispatch optimization %%%%%%%

seconds_in_hour = 3600;
hours_in_day = 24;

water_level_ind = init_water_level_ind; 
day_init = 1;
state = [day_ind(day_init),diff_ind(day_init),water_level_ind];
simul_flow_dispatch = zeros(training_period_days,1);
simul_flow_dispatch_correction = zeros(training_period_days,1);
simul_water_level = zeros(training_period_days,1);
simul_reward = zeros(training_period_days,1);
zero_cell =  zeros(training_period_days,1);
simul_break_lower_level =  zeros(training_period_days,1);
simul_spilling =  zeros(training_period_days,1);
V_table = zeros(length(Q_table(:,1,1,1)),length(Q_table(1,:,1,1)),length(Q_table(1,1,:,1)));
for day = 1:training_period_days
    % Policy:
    if sum(Q_table(state(1),state(2),state(3),:)) == 0
       zero_cell(day) = 1;
       action_ind = randi([1 length(actionDispatch)],1,1);
    else
    [V_table(state(1),state(2),state(3)),action_ind] = max(Q_table(state(1),state(2),state(3),:));
    end
    simul_flow_dispatch(day) = actionDispatch(action_ind); % average hourly (m^3/s)
    revenue = optimRevenue_2018(day_ind(day),action_ind);
    % Water level:
    simul_water_level(day) = stateWaterHeight(state(3));
    net_flow = (inflow(day) - simul_flow_dispatch(day))*seconds_in_hour*hours_in_day; % (m^3)
    delta_height = alpha*net_flow; % (cm)
    next_water_level = stateWaterHeight(state(3)) + delta_height - evap(day_ind(day)); 
    if next_water_level > stateWaterHeight(end) 
        reward = revenue; 
    elseif next_water_level >= stateWaterHeight(1)
        reward = revenue;
    else
        simul_break_lower_level(day) = 1;
        missing_water = (next_water_level - stateWaterHeight(1))*(1/alpha); % (m^3)
        new_daily_total = simul_flow_dispatch(day)*(seconds_in_hour*hours_in_day) + missing_water; % (m^3)
        new_average_dispatch = new_daily_total/(seconds_in_hour*hours_in_day); % (m^3/s)
        [~,corrected_action_ind] = min(abs(actionDispatch-new_average_dispatch));
        simul_flow_dispatch_realized = actionDispatch(corrected_action_ind); % average hourly (m^3/s)
        simul_flow_dispatch_correction(day) = simul_flow_dispatch_realized-simul_flow_dispatch(day);
        water_level_fine = -1e6; 
        next_water_level = stateWaterHeight(1);
        reward = revenue + water_level_fine;
    end
    % (cm) back to (m3/s): 
    cm_to_m3_per_second = 1/(alpha*seconds_in_hour*24);
    simul_spilling(day) = min(0,(stateWaterHeight(end) -  next_water_level)*cm_to_m3_per_second);
    [~,next_water_level_ind] = min(abs(stateWaterHeight-next_water_level));
    % Reward:
    simul_reward(day) = discount_factor(day)*reward;
    % Next state:
    if day == training_period_days
        next_day = 1;
    else
        next_day = day + 1;
    end
    new_state = [day_ind(next_day),diff_ind(next_day),next_water_level_ind]; 
    state = new_state;
end

zeroCells = sum(zero_cell);
waterLevelBreaks = sum(simul_break_lower_level);
totalReward = sum(simul_reward);
           
end