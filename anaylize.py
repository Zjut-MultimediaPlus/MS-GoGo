import matplotlib.pyplot as plt

default_path = './results/LSCIDMR.3layer.bsz_32.sgd5e-05.geo_emb/test.log'
def get_data_seq(path:str=default_path,indicator:str='mAP:'):
    file = open(path)
    lines = file.readlines(100000)
    #print(lines)
    seq = []

    for line in lines:
        start = line.find(indicator)
        if start >= 0 :
            start += len(indicator)
            num = float(line[start:start+5])
            seq.append(num)
    return seq

if __name__ == '__main__':
    have_none = './results/split_16c.cnn.LSCIDMR_16c.2layer.4heads.bsz_64.adam5e-05.plateau_on_map.group_wise_linear_4group_lrelu_2nd/test.log'
    month_path = './results/split_16c.cnn.LSCIDMR_16c.2layer.4heads.bsz_64.adam5e-05.plateau_on_map.use_month.group_wise_linear_4group_lrelu_2nd/test.log'
    loc_path = './results/split_16c.cnn.LSCIDMR_16c.2layer.4heads.bsz_64.adam5e-05.plateau_on_map.use_loc.group_wise_linear_4group_lrelu_2nd/test.log'
    all_path = './results/split_16c.cnn.LSCIDMR_16c.2layer.4heads.bsz_64.adam5e-05.plateau_on_map.use_month.use_loc.group_wise_linear_4group_lrelu_2nd/test.log'

    metric = 'subset_ACC:'
    none_seq = get_data_seq(have_none, metric)[1:101]
    month_seq = get_data_seq(month_path, metric)[1:101]
    loc_seq = get_data_seq(loc_path, metric)[1:101]
    all_seq = get_data_seq(all_path, metric)[1:101]


    print('123'.find('1'))
    plt.figure(figsize=(7,6.7))
    w = 2
    textsize = 20
    plt.plot(range(int(len(none_seq))), none_seq, '#aaaa20', lw=w, label='naked')
    plt.plot(range(int(len(month_seq))), month_seq, '#cc2020', lw=w, label='month')
    plt.plot(range(int(len(loc_seq))), loc_seq, '#20a0a0', lw=w, label='loc')
    plt.plot(range(int(len(all_seq))), all_seq, '#b050b0', lw=w, label='month+loc')
    label_font = {'family':'DejaVu Sans','weight':'normal','size':textsize}
    plt.xlabel('epoch',label_font)
    plt.ylabel(metric[:-1], label_font)
    plt.tick_params(labelsize=12.5)
    #plt.ylim(4,24)
    #plt.ylim(ylim_min, ylim_max)
    plt.legend(prop={'size':textsize})
    plt.savefig('./fig/converge_'+metric[:-1]+'.png', dpi=600)
    plt.show()