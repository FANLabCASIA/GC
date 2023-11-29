function MPC_calculate_HCP_R(i)
    hemi='R';
    disp(i);
    disp(hemi);
    list = './HCP_twins.txt';
    SubID = textread(list,'%s');
    sub = SubID{i};
    disp(sub);

    % check whether the mpc is exist!
    if exist(strcat('/n01dat01/dyli/data4n02/HCP_twin/',sub, '/', sub, '_', hemi, '_sc_MPC.mat'), 'file')==0
        path = strcat('/n01dat01/dyli/data4n02/HCP_twin/',sub, '/vertex_volume_sc_', hemi, '.txt');
        if exist(path, 'file')==0
            disp(sub);
            disp('no data');
        else
            data = load(path);
            disp(size(data));
            I = transpose(data);
            R = partialcorr(I, mean(I,2));
            display([num2str(sum(sum(1*(isnan(R)))))]);
            save(strcat('/n01dat01/dyli/data4n02/HCP_twin/', sub, '/', sub, '_', hemi, '_sc_MPC.mat'), 'R');
        end
    end

    % remove the sc txt data
    if exist(strcat('/n01dat01/dyli/data4n02/HCP_twin/',sub, '/', sub, '_', hemi, '_sc_MPC.mat'), 'file')
        disp(strcat('the TC is finished!', sub))
        delete(path)
    end
end