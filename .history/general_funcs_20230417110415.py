import torch
from explorer import find_approx_func_under_emax, fix_hidden_layer_approx
from func_sample import construct_batches, FuncSample
from approximator import UnaryApproximator, PolyApproximator
from domain import Domain
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math


def train(func, func_name, num, domain, batches, sample_num):
    info = {
        'e_max': 0,  # no early break
        'save': True,
        'batch_num': 100000,
        'optimize_level': 3,
        'max_iteration': 10000,
        'lr': 1e-2,
        'stage2_lr': 1e-5,
        'stage2_error': 0.01,
        'test_step': 0.01,
        'find_init_approx_time': 200,
        'ua_adjust': 0.5,
        'print_log': True,
        'cuda': True
    }

    domain = Domain(-10, 10)
    e = fix_hidden_layer_approx(func, domain, info, batches, sample_num, func_name, num)
    return e


def train_binary(func, func_name, num, domain, batches, sample_num):
    info = {
        'e_max': 0,
        'save': True,
        'batch_num': 100000,
        'optimize_level': 3,
        'max_iteration': 100000,
        'lr': 2e-3,
        'stage2_lr': 1e-4,
        'stage2_error': 0.1,
        'test_step': 0.05,
        'find_init_approx_time': 1000,
        'print_log': True,
        'ua_adjust': 1,
        'cuda': True
    }

    e = fix_hidden_layer_approx(func, domain, info, batches, sample_num, func_name, num)
    return e


def train_tenary(func, func_name, num, domain, batches, sample_num):
    info = {
        'e_max': 0,
        'save': True,
        'batch_num': 100000,
        'optimize_level': 3,
        'max_iteration': 50000,
        'lr': 1e-3,
        'stage2_lr': 1e-4,
        'stage2_error': 0.1,
        'test_step': 0.5,
        'find_init_approx_time': 50,
        'print_log': True,
        'ua_adjust': 1,
        'cuda': True
    }

    e = fix_hidden_layer_approx(func, domain, info, batches, sample_num, func_name, num)
    return e


def train_funcs():
    func = torch.cos
    func_name = 'cos'
    e_list = []
    domain = Domain(-10, 10)
    batches, sample_num = construct_batches(func, domain, 0.01, 100000)
    for num in [16]:
        e = train(func, func_name, num, domain, batches, sample_num)
        e_list.append(e.item())
    print(e_list)


def train_binary_funcs():
    func = lambda x, y: torch.sin(x + y)
    func_name = 'exp_exp2'
    e_list = []
    domain = Domain((-9.99, -9.99), (10, 10))
    batches, sample_num = construct_batches(func, domain, 0.05, 100000)
    for num in [64]:
        e = train_binary(func, func_name, num, domain, batches, sample_num)
        e_list.append(e.item())
    print(e_list)


def train_tenary_funcs():
    func = lambda x, y, z: torch.exp(x) + torch.exp(y) + torch.exp(z)
    func_name = 'exp_exp_exp'
    e_list = []
    domain = Domain((-5, -5, -5), (5, 5, 5))
    batches, sample_num = construct_batches(func, domain, 0.5, 100000)
    for num in [64]:
        e = train_tenary(func, func_name, num, domain, batches, sample_num)
        e_list.append(e.item())
    print(e_list)


def cal_error():
    func = lambda x: x * x
    domain = Domain(-10, 10)
    nums = range(64)
    e_list = []
    for num in nums:
        approx = UnaryApproximator(func, num, domain, 3, cuda=False)
        approx.load_state_dict(torch.load('funcs\\x^2_' + str(num) + '.pkl'))
        func_sample = FuncSample(func, domain, 0.01, False)
        e = sum(torch.sum(torch.abs((y - approx(x)) / (y))) for x, y in func_sample) / len(func_sample)
        e_list.append(e.item())
    print(e_list)


def cal_binary_error():
    func = lambda x, y: torch.sin(x + y)
    domain = Domain((-9.99, -9.99), (10, 10))
    nums = range(64)
    e_list = []
    for num in [64]:
        approx = PolyApproximator(lambda x, y: x + y, 64, domain)
        approx.load_state_dict(torch.load('funcs\\exp_exp2_' + str(num) + '.pkl'))

        func_sample = FuncSample(func, domain, 0.05, False)
        xs = [x for x, y in func_sample]
        ys = [y for x, y in func_sample]
        approxs = [approx(x.view(-1, 2)) for x in xs]
        print('relative: ', (sum(abs((y - y_) / y) for y, y_ in zip(ys, approxs)) / len(approxs)).item())
        loss_fn = torch.nn.MSELoss(reduction='mean')
        print('MSE: ', loss_fn(torch.tensor(approxs), torch.tensor(ys)).item())
    #     e_list.append(e.item())
    # print(e_list)


def draw_cost_precision():
    sin_e = [1.058646321296692, 0.248815655708313, 0.13380570709705353, 0.04738518223166466,
             0.025365902110934258, 0.021470624953508377, 0.0175121258944273, 0.012452647089958191,
             0.010071536526083946, 0.007508957292884588, 0.006554802414029837, 0.005205216351896524,
             0.004648460540920496, 0.003923114854842424, 0.00356465601362288, 0.0030953208915889263,
             0.0029485500417649746, 0.0025021692272275686, 0.0022862518671900034, 0.002082702936604619,
             0.001898818532936275, 0.0016324121970683336, 0.0017026190180331469, 0.0014808629639446735,
             0.0014482794795185328, 0.0013223218265920877, 0.0012880854774266481, 0.0014956040540710092,
             0.0012546052457764745, 0.0010467087849974632, 0.0009755578357726336, 0.0009048302890732884]
    exp_e = [14.520381927490234, 0.20706996321678162, 0.09535159915685654, 0.05354217439889908,
             0.03398491442203522, 0.023400288075208664, 0.017064057290554047, 0.012987052090466022,
             0.01020196545869112, 0.008226596750319004, 0.006772249937057495, 0.005671539343893528,
             0.004818914458155632, 0.004141902085393667, 0.003597995964810252, 0.003154564183205366,
             0.002793592866510153, 0.0024805099237710238, 0.0022258106619119644, 0.0020070660393685102,
             0.0018129694508388638, 0.0016500360798090696, 0.0015084492042660713, 0.0013851625844836235,
             0.0012741194805130363, 0.0011772731086239219, 0.0010900709312409163, 0.0010087265400215983,
             0.0009428408229723573, 0.0008813372696749866, 0.0008250821265392005, 0.000772425381001085]
    tan_e = [1.0758872032165527, 1.541081190109253, 0.7751335501670837, 0.4592529237270355,
             0.5198279023170471, 0.30281105637550354, 0.2302623838186264, 0.20596107840538025,
             0.18802762031555176, 0.15636755526065826, 0.14517955482006073, 0.13369230926036835,
             0.12017563730478287, 0.11026418954133987, 0.1030397117137909, 0.10052993148565292,
             0.0920042172074318, 0.08770705759525299, 0.08191324025392532, 0.07723717391490936,
             0.07270139455795288, 0.07692435383796692, 0.07308939844369888, 0.06391710788011551,
             0.06470116227865219, 0.06244484707713127, 0.06085273250937462, 0.06156335771083832,
             0.05399680510163307, 0.055418796837329865, 0.051158539950847626, 0.057352080941200256]
    rep_e = [0.28027209639549255, 0.11760008335113525, 0.0740254670381546, 0.05548398196697235,
             0.0449991412460804, 0.038458146154880524, 0.0341058224439621, 0.02984168380498886,
             0.0267020296305418, 0.024383431300520897, 0.022200411185622215, 0.02010412886738777,
             0.01985490880906582, 0.017172444611787796, 0.01650516502559185, 0.015530543401837349,
             0.015470271930098534, 0.014294764026999474, 0.014806711114943027, 0.012799779884517193,
             0.011790558695793152, 0.01076316274702549, 0.010347607545554638, 0.011083433404564857,
             0.010715339332818985, 0.010558927431702614, 0.009945673868060112, 0.00967763178050518,
             0.009431049227714539, 0.008390406146645546, 0.008270293474197388, 0.00895445141941309]

    x2_e = [0.5250117182731628, 0.2220403105020523, 0.13704292476177216, 0.10042320936918259,
            0.10251478105783463, 0.08040712028741837, 0.05642220005393028, 0.05137740448117256,
            0.06673480570316315, 0.03885871544480324, 0.058027151972055435, 0.03627215325832367,
            0.040279727429151535, 0.024990953505039215, 0.05318906903266907, 0.019152715802192688,
            0.05055951327085495, 0.025982100516557693, 0.04236174002289772, 0.04012153670191765,
            0.03790038824081421, 0.05265270173549652, 0.07115954160690308, 0.115602508187294,
            0.01176951453089714, 0.02318509668111801, 0.03256027400493622, 0.0170428603887558,
            0.02697555348277092, 0.017426850274205208, 0.04416557401418686, 0.07812013477087021]

    sqrt_e = [0.04203234985470772, 0.013533483259379864, 0.0062464275397360325, 0.0033819167874753475,
              0.0020827611442655325,
              0.0015097964787855744, 0.001135779544711113, 0.0008888808661140501, 0.0007444811053574085,
              0.0006625506212003529, 0.0005970387137494981, 0.0005466882139444351, 0.00047564992564730346,
              0.0004008675168734044, 0.00040748846367932856, 0.00038093997864052653, 0.00036348894354887307,
              0.00035103491973131895, 0.00034720965777523816, 0.0002831385936588049, 0.0002605659537948668,
              0.000244483002461493, 0.00023888281430117786, 0.00023630277428310364, 0.00023261569731403142,
              0.00021028191258665174, 0.0002024088316829875, 0.00021082416060380638, 0.00020476699864957482,
              0.00019678038370329887, 0.00019379086734261364, 0.00017610359645914286]

    nums = range(4, 132, 4)

    # To excel
    # import xlwt
    # workbook = xlwt.Workbook(encoding='utf-8')
    # worksheet = workbook.add_sheet('单元函数的隐层与误差')
    # all_es = [nums, sin_e, exp_e, tan_e, rep_e, x2_e, sqrt_e]
    # names = ['', 'sin(x)', 'tan(x)', 'exp(x)', '1/x', 'x^2', 'sqrt(x)']
    # for i, name in enumerate(names):
    #     worksheet.write(i, 0, label=name)
    #     for j, value in enumerate(all_es[i]):
    #         worksheet.write(i, j + 1, label=value)
    # workbook.save('unary_funcs.xls')

    plt.figure()
    plt.plot(nums, sin_e, label='sin')
    plt.plot(nums, exp_e, label='exp')
    plt.plot(nums, tan_e, label='tan')
    plt.plot(nums, rep_e, label='1/x')
    plt.plot(nums, x2_e, label='x^2')
    plt.plot(nums, sqrt_e, label='sqrt')
    plt.legend()
    # alpha = 0.3
    # plt.plot(nums, [alpha * nums[i] + (1 - alpha) * (losses[i]) for i in range(len(nums))])
    plt.yscale('log')
    x_major_locator = MultipleLocator(8)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid()
    plt.show()


def cal_MSE():
    func = lambda x: torch.exp(x)
    domain = Domain(0, 10)
    nums = range(4, 132, 4)
    e_list = []
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for num in [64]:
        approx = UnaryApproximator(func, num, domain, 3, cuda=False)
        approx.load_state_dict(torch.load('funcs\\exp2_' + str(num) + '.pkl'))
        func_sample = FuncSample(func, domain, 0.01, False)
        xs = [x for x, y in func_sample]
        ys = [y for x, y in func_sample]
        approx = [approx(x) for x in xs]
        e = loss_fn(torch.tensor(approx), torch.tensor(ys))
        e_list.append(e.item())
    print(e_list)


def record_MSE():
    sin_e = [0.4886060655117035, 0.028690176084637642, 0.01226082257926464, 0.0020463326945900917,
             0.0005225809291005135, 0.0003912397369276732, 0.0002194855478592217, 0.0001070663274731487,
             7.240290142362937e-05, 4.259403795003891e-05, 3.09897841361817e-05, 2.0916389985359274e-05,
             1.4629456018155906e-05, 1.1380716387066059e-05, 8.963058462541085e-06, 6.8281728999863844e-06,
             5.577353476837743e-06, 4.360308139439439e-06, 3.5333137020643335e-06, 2.8162930902908556e-06,
             2.3253826384461718e-06, 1.9505682757881004e-06, 1.6532571862626355e-06, 1.3876182265448733e-06,
             1.1708580132108182e-06, 1.0036278581537772e-06, 8.814387228994747e-07, 7.823498435755027e-07,
             6.635885938521824e-07, 5.733647867600666e-07, 5.008231482861447e-07, 4.343459352185164e-07]

    tan_e = [1931.2445068359375, 1928.4461669921875, 1928.98876953125, 1928.6903076171875, 1957.231689453125,
             1929.68505859375, 1928.6617431640625, 1934.22216796875, 1924.942138671875, 1926.9107666015625,
             1929.7977294921875, 1921.9276123046875, 1924.185302734375, 1925.456787109375, 1917.4912109375,
             1921.275390625, 1923.03759765625, 1911.964599609375, 1917.8333740234375, 1918.84326171875, 1899.5625,
             1929.1873779296875, 1918.88916015625, 1925.781982421875, 1913.96875, 1916.5115966796875,
             1922.3145751953125, 1910.2557373046875, 1913.9891357421875, 1917.2340087890625, 1905.6793212890625,
             1910.86083984375]

    exp_e = [17040910.0, 2525272.5, 340054.59375, 86193.234375, 31341.947265625, 13247.1630859375,
             7083.6826171875, 4140.19140625, 2423.887451171875, 1549.7532958984375, 1022.5396118164062,
             716.1115112304688, 526.3041381835938, 377.8879699707031, 277.1393737792969, 223.76583862304688,
             171.7803497314453, 143.63955688476562, 106.81912994384766, 92.43018341064453, 81.62174987792969,
             58.125892639160156, 52.369468688964844, 47.851036071777344, 42.530399322509766, 30.915071487426758,
             25.98383140563965, 24.494394302368164, 21.52037811279297, 21.33582878112793, 19.787784576416016,
             13.018966674804688]

    rep_e = [16.37274742126465, 16.253467559814453, 16.11957550048828, 16.013883590698242, 15.925211906433105,
             15.834794044494629, 15.747502326965332, 15.637561798095703, 15.53161334991455, 15.428796768188477,
             15.318893432617188, 15.20374584197998, 15.119362831115723, 14.988825798034668, 14.920109748840332,
             14.794144630432129, 14.699884414672852, 14.58439826965332, 14.511353492736816, 14.300675392150879,
             14.292722702026367, 14.108880996704102, 14.046426773071289, 13.91042709350586, 13.852605819702148,
             13.794172286987305, 13.61573600769043, 13.610252380371094, 13.600226402282715, 13.366630554199219,
             13.277355194091797, 13.276175498962402]

    x2_e = [21.801193237304688, 0.6527703404426575, 0.1029185801744461, 0.029346344992518425, 0.01148395799100399,
            0.0052690752781927586, 0.002646234817802906, 0.0015327800065279007, 0.0009101128089241683,
            0.000636641460005194, 0.0005011080065742135, 0.00033084521419368684, 0.0002791748265735805,
            0.00025342637673020363, 0.0003151718119625002, 0.0002917479432653636, 0.00041403688373975456,
            0.0006790357874706388, 0.001201477600261569, 0.0022720370907336473, 0.004808178171515465,
            0.006308203563094139, 0.007935128174722195, 0.010569855570793152, 0.010597213171422482,
            0.011135865934193134, 0.011615786701440811, 0.012799986638128757, 0.014669124037027359, 0.01377659197896719,
            0.014891698025166988, 0.014828063547611237]

    sqrt_e = [0.003844654420390725, 0.0003981454356107861, 8.726256055524573e-05, 2.5176983399433084e-05,
              1.0250744708173443e-05, 5.210583367443178e-06, 3.519032816257095e-06, 2.1864093469048385e-06,
              1.5583356116621871e-06, 1.2596474334714003e-06, 1.0341599363528076e-06, 9.070481041817402e-07,
              6.719928933307528e-07, 4.869644953942043e-07, 4.892708602710627e-07, 4.463159939405159e-07,
              3.966197823501716e-07, 3.768471685816621e-07, 3.6654026303040155e-07, 2.516585482226219e-07,
              2.302402748455279e-07, 2.20921066329538e-07, 2.0031303904488595e-07, 1.9386150995615026e-07,
              1.902224653349549e-07, 1.6740234798362508e-07, 1.603228270141699e-07, 1.6028232607823156e-07,
              1.5528452479429689e-07, 1.5630249095011095e-07, 1.3603795423478005e-07, 1.3199837667343672e-07]

    # To excel
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('单元函数的隐层与误差')
    nums = range(4, 132, 4)
    all_es = [nums, sin_e, tan_e, exp_e, rep_e, x2_e, sqrt_e]
    names = ['', 'sin(x)', 'tan(x)', 'exp(x)', '1/x', 'x^2', 'sqrt(x)']
    for i, name in enumerate(names):
        worksheet.write(i, 0, label=name)
        for j, value in enumerate(all_es[i]):
            worksheet.write(i, j + 1, label=value)
    workbook.save('MSE.xlsx')

def draw_approx_func():
    func = torch.tan
    nums = [4, 8, 16, 32]
    errors = []
    plt.figure()
    count = 0
    func_sample = FuncSample(func, Domain(-10, 10), 0.01, False)
    xs = [x for x, y in func_sample]
    ys = [y for x, y in func_sample]
    for i in nums:
        count += 1
        approx = UnaryApproximator(func, i, Domain(-10, 10), 3, cuda=False)
        approx.load_state_dict(torch.load('funcs\\tan_' + str(i) + '.pkl'))
        plt.subplot(1, 4, count)
        plt.plot(xs, ys)
        plt.plot(xs, [approx(x) for x in xs])
        plt.ylim([-8, 8])
        plt.title(str(i))
        plt.grid()

    plt.show()


train_funcs()