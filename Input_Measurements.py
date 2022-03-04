#!/usr/bin/env python
# coding: utf-8


from astropy import constants as const

mearth = const.M_earth.cgs.value  # grams
msun = const.M_sun.cgs.value

mstar = 1.034

measured = [[2455030.145908, 2455043.95822, 2455057.653172, 2455071.596735, 2455085.38127, 2455099.264417,
             2455113.185062, 2455126.92793, 2455140.770799, 2455168.423202, 2455196.175606, 2455210.063613,
             2455223.940509, 2455237.764627, 2455251.647079, 2455265.49967, 2455279.43351, 2455293.318046,
             2455307.202581, 2455321.07531, 2455334.919568, 2455348.776324, 2455362.588637, 2455376.50095,
             2455404.131825, 2455417.997611, 2455473.228112, 2455487.175842, 2455500.964543, 2455514.8088,
             2455528.635697, 2455542.452176, 2455570.119857, 2455583.923836, 2455597.813927, 2455611.625546,
             2455625.514941, 2455653.16179, 2455667.090074, 2455680.989888, 2455694.834145, 2455708.845763,
             2455722.57752, 2455736.445388, 2455750.363256, 2455764.242237, 2455778.031632, 2455791.993945,
             2455805.725008, 2455819.641487, 2455847.305697, 2455861.177037, 2455875.031017, 2455888.843329,
             2455902.65842, 2455916.404066, 2455930.403879, 2455944.137719, 2455957.950727, 2455971.744984,
             2455985.604519, 2455999.409887, 2456054.826499, 2456068.682563, 2456082.576125, 2456096.365521,
             2456110.329223, 2456151.891162, 2456165.736807, 2456179.740788, 2456193.452406, 2456207.302912,
             2456221.241615, 2456235.095594, 2456262.723692, 2456276.604754, 2456290.485817, 2456331.9672,
             2456345.819791, 2456359.516826, 2456373.473583, 2456387.257424, 2456401.154458],
            [2454955.917878, 2454972.152271, 2454988.406663, 2455004.642792, 2455020.892532, 2455037.141229,
             2455053.392914, 2455069.647723, 2455085.882879, 2455102.129216, 2455118.373053, 2455134.611334,
             2455150.863088, 2455167.1013, 2455199.525363, 2455215.748505, 2455248.185276, 2455264.399113,
             2455280.577672, 2455296.801926, 2455313.028887, 2455329.250502, 2455345.471839, 2455361.690328,
             2455377.896249, 2455394.138488, 2455410.387117, 2455426.637343, 2455442.87868, 2455459.128142,
             2455475.372811, 2455491.62026, 2455507.874791, 2455540.371215, 2455572.852361, 2455589.095226,
             2455605.326771, 2455621.561649, 2455654.005851, 2455670.22573, 2455686.436163, 2455702.657431,
             2455718.875573, 2455735.096633, 2455751.280469, 2455767.508195, 2455783.735921, 2455799.956424,
             2455816.195053, 2455832.427987, 2455848.63863, 2455864.890939, 2455881.146443, 2455913.643143,
             2455929.896564, 2455946.135609, 2455962.383613, 2455978.633561, 2456011.136373, 2456027.371461,
             2456043.610367, 2456059.831078, 2456076.058874, 2456092.290072, 2456108.517381, 2456140.934569,
             2456157.151669, 2456173.382868, 2456189.593302, 2456221.998267, 2456254.473302, 2456270.702486,
             2456286.946948, 2456303.183771, 2456335.660195, 2456351.906046, 2456368.160994, 2456384.398788,
             2456400.659292]]

epoch = [[5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 37, 38, 39,
          40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69,
          70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 99, 100, 101, 102, 103,
          104],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62,
          63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 78, 80, 81, 82, 83, 85, 86, 87, 88, 89]]

error = [[0.01666667, 0.01597222, 0.03333333, 0.02916667, 0.02708333, 0.025, 0.01527778, 0.02152778, 0.00902778,
          0.03194444, 0.02013889, 0.03055556, 0.02152778, 0.02152778, 0.01805556, 0.01597222, 0.02222222, 0.02152778,
          0.01805556, 0.01388889, 0.02361111, 0.025, 0.03333333, 0.02013889, 0.02986111, 0.01041667, 0.01666667,
          0.02361111, 0.02916667, 0.0125, 0.02222222, 0.02430556, 0.00833333, 0.02361111, 0.03055556, 0.01875,
          0.01597222, 0.03125, 0.03680556, 0.01527778, 0.01736111, 0.01319444, 0.01180556, 0.02708333, 0.02152778,
          0.01944444, 0.02222222, 0.02222222, 0.02222222, 0.02291667, 0.02222222, 0.02222222, 0.02152778, 0.01597222,
          0.03333333, 0.02291667, 0.02638889, 0.02152778, 0.01319444, 0.02916667, 0.02986111, 0.01458333, 0.01875,
          0.01527778, 0.01736111, 0.025, 0.01944444, 0.02013889, 0.02083333, 0.02708333, 0.01736111, 0.01458333,
          0.01875, 0.02361111, 0.02916667, 0.04027778, 0.01388889, 0.0125, 0.02152778, 0.02152778, 0.03055556,
          0.03402778, 0.0125],
         [0.00402778, 0.00256945, 0.00444445, 0.00319445, 0.00361111, 0.00340278, 0.00270833, 0.00263889, 0.00263889,
          0.00298611, 0.003125, 0.00347222, 0.00284722, 0.00354167, 0.00291667, 0.00319444, 0.00291667, 0.00388889,
          0.00333333, 0.00284722, 0.00361111, 0.00243056, 0.00326389, 0.00368056, 0.00298611, 0.00340278, 0.00347222,
          0.003125, 0.00263889, 0.00298611, 0.00319444, 0.00305556, 0.00347222, 0.00305556, 0.00347222, 0.003125,
          0.00319444, 0.00381944, 0.00347222, 0.003125, 0.00305556, 0.00256944, 0.00326389, 0.00388889, 0.00333333,
          0.00305556, 0.00270833, 0.00354167, 0.00395833, 0.00291667, 0.003125, 0.003125, 0.00333333, 0.003125,
          0.00326389, 0.00375, 0.00326389, 0.00305556, 0.00305556, 0.00340278, 0.00333333, 0.00291667, 0.00263889,
          0.003125, 0.00298611, 0.00340278, 0.003125, 0.003125, 0.00298611, 0.00284722, 0.00361111, 0.00284722,
          0.00319444, 0.00333333, 0.00277778, 0.00354167, 0.003125, 0.00361111, 0.00347222]]

# In[ ]:

orbital_elements_3body = {'mass_b': [mearth / msun, 10 * mearth / msun],
                          'period_b': [13.849 - .05, 13.849 + .05],
                          'eccentricity_b': [0.00, 0.04],
                          'inclination_b': [90],
                          'longnode_b': [-45.0, 45.0],
                          'argument_b': [0.0, 360.0],
                          'mean_anomaly_b': [0.0, 360.0],

                          'mass_c': [mearth / msun, 10 * mearth / msun],
                          'period_c': [16.2369 - .05, 16.2369 + .05],
                          'eccentricity_c': [0.00, 0.04],
                          'inclination_c': [90],
                          'longnode_c': [0.0],
                          'argument_c': [0.0, 360.0],
                          'mean_anomaly_c': [0.0, 360.0]}

# orbital_elements_4body = {'mass_b': [mearth / msun, 10 * mearth / msun],
#                           'period_b': [13.849 - .05, 13.849 + .05],
#                           'eccentricity_b': [0.00, 0.04],
#                           'inclination_b': [90],
#                           'longnode_b': [-45.0, 45.0],
#                           'argument_b': [0.0, 360.0],
#                           'mean_anomaly_b': [0.0, 360.0],
#
#                           'mass_c': [mearth / msun, 10 * mearth / msun],
#                           'period_c': [16.2369 - .05, 16.2369 + .05],
#                           'eccentricity_c': [0.00, 0.04],
#                           'inclination_c': [90],
#                           'longnode_c': [0.0],
#                           'argument_c': [0.0, 360.0],
#                           'mean_anomaly_c': [0.0, 360.0]}

theta_max_2_mil = {'mass_b': [1.30930463e-05],
                   'period_b': [1.38386888e+01],
                   'eccentricity_b': [3.09587269e-02],
                   'inclination_b': [8.96422247e+01],
                   'longnode_b': [-6.87326187e-01],
                   'argument_b': [8.68035125e+01],
                   'mean_anomaly_b': [2.21200549e+02],

                   'mass_c': [2.26384959e-05],
                   'period_c': [1.62402849e+01],
                   'eccentricity_c': [4.48114084e-03],
                   'inclination_c': [9.01372655e+01],
                   'longnode_c': [0.0],
                   'argument_c': [1.10617410e+02],
                   'mean_anomaly_c': [3.30816830e+02]}
theta_max_1mil = [1.31527917e-05, 1.38383552e+01, 2.55863640e-02, 8.97262424e+01,
                  -9.42715184e-01, 8.32508952e+01, 2.24596454e+02, 2.30641953e-05,
                  1.62404163e+01, 1.01459431e-03, 9.04586424e+01, 2.46597087e+02,
                  1.94689766e+02]

theta_max_2mil = [1.32881509e-05, 1.38384545e+01, 2.69930875e-02, 8.95997194e+01,
                  -1.01301224e+00, 8.34941798e+01, 2.24348768e+02, 2.29997664e-05,
                  1.62404535e+01, 5.74049502e-04, 9.03723667e+01, 1.14143981e+02,
                  3.27143499e+02]

orbital_elements_1_no_ecc = {'mass_b': [mearth / msun, 100 * mearth / msun],
                             'period_b': [165., 175.],
                             'eccentricity_b': [0.00],
                             'inclination_b': [90.0],
                             'longnode_b': [0.0],
                             'argument_b': [0.0],
                             'mean_anomaly_b': [0.0, 360.0],
                             'mass_c': [mearth / msun, 100 * mearth / msun],
                             'period_c': [165., 175.],
                             'eccentricity_c': [0.00],
                             'inclination_c': [90.0],
                             'longnode_c': [0.0],
                             'argument_c': [0.0],
                             'mean_anomaly_c': [0.0, 360.0]}

donghong_elements = [4.6813098187, 13.8365700314, 0.0250513232, 90.4221451464, 0.3013433398, 83.1024519679,
                     209.2594293527], [7.9316353360, 16.2369403451, 0.0061762733, 90.2377925833, 0.0000000000,
                                       247.5376924170, 165.7005024527]

# mass, period, eccentricity, inclination, longnode, argument, mean_anomaly

orbital_elements_4body = {'mass_b': [mearth / msun, 10 * mearth / msun],
                          'period_b': [13.849 - .05, 13.849 + .05],
                          'eccentricity_b': [0.00, 0.04],
                          'inclination_b': [90],
                          'longnode_b': [-45.0, 45.0],
                          'argument_b': [0.0, 360.0],
                          'mean_anomaly_b': [0.0, 360.0],

                          'mass_c': [mearth / msun, 10 * mearth / msun],
                          'period_c': [16.2369 - .05, 16.2369 + .05],
                          'eccentricity_c': [0.00, 0.04],
                          'inclination_c': [90],
                          'longnode_c': [0.0],
                          'argument_c': [0.0, 360.0],
                          'mean_anomaly_c': [0.0, 360.0],

                          'mass_d': [mearth / msun, 200 * mearth / msun],
                          'period_d': [40, 300],
                          'eccentricity_d': [0.00],
                          'inclination_d': [90.0],
                          'longnode_d': [0.0],
                          'argument_d': [0.0],
                          'mean_anomaly_d': [0.0, 360.0]}

# 'mass_d'         : [mearth/msun,  10 * mearth/msun],
# 'period_d'       : [165.,         175.      ],
# 'eccentricity_d' : [0.00,         0.04             ],
# 'inclination_d'  : [0,            180             ],
# 'longnode_d'     : [-45.0,        45.0             ],
# 'argument_d'     : [0.0,          360.0            ],
# 'mean_anomaly_d' : [0.0,          360.0            ]}


orbital_elements_5body = {'mass_b': [mearth / msun, 10 * mearth / msun],
                          'period_b': [13.849 - .05, 13.849 + .05],
                          'eccentricity_b': [0.00, 0.04],
                          'inclination_b': [90],
                          'longnode_b': [-45.0, 45.0],
                          'argument_b': [0.0, 360.0],
                          'mean_anomaly_b': [0.0, 360.0],

                          'mass_c': [mearth / msun, 10 * mearth / msun],
                          'period_c': [16.2369 - .05, 16.2369 + .05],
                          'eccentricity_c': [0.00, 0.04],
                          'inclination_c': [90],
                          'longnode_c': [0.0],
                          'argument_c': [0.0, 360.0],
                          'mean_anomaly_c': [0.0, 360.0],

                          'mass_d': [mearth / msun, 200 * mearth / msun],
                          'period_d': [40, 300],
                          'eccentricity_d': [0.00],
                          'inclination_d': [90.0],
                          'longnode_d': [0.0],
                          'argument_d': [0.0],
                          'mean_anomaly_d': [0.0, 360.0],

                          'mass_e': [mearth / msun, 200 * mearth / msun],
                          'period_e': [40, 300],
                          'eccentricity_e': [0.00],
                          'inclination_e': [90.0],
                          'longnode_e': [0.0],
                          'argument_e': [0.0],
                          'mean_anomaly_e': [0.0, 360.0]}

# In[ ]:

# BJD, MNVEL, ERRVEL
rv = [[2456138.960111, 2456202.74719, 2456880.986613, 2456892.948262, 2456909.872368, 2456912.827319, 2457151.077157,
       2457180.097595, 2457208.096426, 2457232.984981, 2457243.082964, 2458294.878483, 2458322.919588, 2458329.871325,
       2458337.067316, 2458383.960506, 2458391.919851, 2458395.857491, 2458723.006589],
      [11.9236788815611, -16.1134940581337, -4.06925772245635, -6.54617470079044, -5.7705062430896, -1.8848456824408,
       7.81277266544407, 0.0602771570512033, 8.25316787388147, -6.29082298836888, -15.3491948450888, 7.00556500279268,
       -2.28990086574453, 10.7279022469152, -9.31859521574567, 0.983863725849822, 7.25795239603468, 7.04186679076109,
       4.2862555505861],
      [3.18929839134216, 4.51967430114746, 3.90615701675415, 3.28753995895386, 3.43545699119568, 3.58018112182617,
       3.54026436805725, 3.5506637096405, 3.62212514877319, 3.50754976272583, 4.02697372436523, 3.27114105224609,
       4.85352230072021, 3.48017978668213, 3.79149770736694, 4.11460399627686, 3.638827085495, 3.71935081481934,
       3.51347351074219]]

rv_102021 = [[2456138.960111, 2456202.74719, 2456880.986613, 2456892.948262, 2456909.872368, 2456912.827319,
              2457151.077157, 2457180.097595, 2457208.096426, 2457232.984981, 2457243.082964, 2458294.878483,
              2458322.919588, 2458329.871325, 2458337.067316, 2458383.960506, 2458391.919851, 2458395.857491,
              2458723.006589, 2459478.904109, 2459481.905374, 2459484.862818, 2459489.803662, 2459497.749138,
              2459498.86784],
             [12.404249810853, -14.5963141891602, -3.47631548460235, -4.73971625662051, -4.73468194423096,
              -1.9966392005072, 5.93742489768356, -0.425009302275541, 9.76505923893274, -5.46596337909981,
              -13.4942820872152, 6.23869498097261, -0.946008986397269, 9.99261905690156, -8.29972485381556,
              1.59233159057487, 7.79636082371804, 8.37888837501965, 3.68386120816361, 4.31192567985581,
              -0.0409648147076069, -0.395656064742052, -18.7982185582625, -5.07544385997046, 8.89878430687106],
             [3.21964406967163, 4.39822244644165, 4.03610754013062, 3.29025220870972, 3.24858331680298,
              3.47713494300842, 3.54778790473938, 3.53474259376526, 3.71352291107178, 3.42335081100464,
              4.02084493637085, 3.31546902656555, 4.99050235748291, 3.52835369110107, 3.82774949073792, 4.1320366859436,
              3.7557065486908, 3.60283136367798, 3.57544469833374, 3.66722750663757, 3.53560853004456, 3.37570691108704,
              3.68359160423279, 3.40241837501526, 3.40831613540649]]
