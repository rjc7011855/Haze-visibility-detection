#import matplotlib.pyplot as plt
import xlwt
import numpy as np
trust = [1944.7827, 1981.0858, 2017.389, 2001.2683, 1985.1477, 1981.0444, 1976.9412, 1976.9412, 1976.9412, 1930.9658, 1884.9904, 1944.427, 2003.8636, 2006.6001, 2009.3364, 2005.3422, 2001.3479, 2028.8873, 2056.4268, 2031.6168, 2006.8069, 1968.1443, 1929.4818, 1945.106, 1960.7301, 1939.3024, 1917.8748, 1929.3556, 1940.8363, 1967.7531, 1994.6697, 1952.0605, 1909.4513, 1917.3304, 1925.2096, 1923.083, 1920.9563, 1918.647, 1916.3376]
resnet = [1849.9849, 1844.7361, 1844.9531, 1841.2848, 1845.4106, 1847.4423, 1850.8181, 1849.2114, 1845.6809, 1851.8253, 1839.2051, 1842.7566, 1852.7578, 1845.261, 1849.2424, 1851.0542, 1854.7749, 1849.4438, 1852.9309, 1855.3002, 1856.0449, 1860.1937, 1855.4883, 1853.246, 1853.106, 1850.6216, 1851.9507, 1849.4795, 1857.0707, 1858.0596, 1858.8015, 1857.6897, 1856.0503, 1852.9463, 1850.0609, 1847.6769, 1861.7646, 1860.1891, 1858.8744]
diracnet = [2042.5166, 2059.426, 2046.9159, 2075.1738, 2069.1936, 2033.6383, 2062.3203, 2058.9299, 2064.6353, 2065.2898, 2058.4114, 2075.2761, 2067.1721, 2056.8428, 2057.0205, 2052.6235, 2077.7061, 2069.7266, 2078.8831, 2070.2463, 2049.0286, 2067.9128, 2047.9785, 2064.7917, 2042.278, 2051.9844, 2041.3546, 2058.2776, 2048.5845, 2035.7628, 2069.6187, 2056.854, 2053.3684, 2084.3259, 2063.5601, 2021.0168, 2045.7816, 2049.2537, 2030.3386]
new_diracnet = [1945.7632, 2071.4236, 2058.199, 2071.7312, 2047.4913, 2052.3425, 2009.3336, 2058.498, 2087.1716, 1965.9138, 2145.4763, 2005.8306, 1904.3922, 1981.9802, 2002.5332, 1950.5007, 1897.9772, 1897.9603, 1918.6686, 1880.2371, 1898.7393, 1873.2694, 1876.8097, 1930.9885, 1922.7616, 1933.6606, 1889.5044, 1924.3745, 1891.9723, 1889.9818, 1896.087, 1906.6168, 1921.8329, 1977.4788, 2019.8735, 2059.9106, 2027.9332, 2001.6265, 2056.5996]
trust = np.array(trust)/10
resnet = np.array(resnet)/10
diracnet = np.array(diracnet)/10
new_diracnet = np.array(new_diracnet)/10
f1 = xlwt.Workbook()
sheet1 = f1.add_sheet("3netpredict")
xlsx_dir = "./xlsx/"
sheet1.write(0, 0, "Trust")
sheet1.write(0, 1, "ResNet")
sheet1.write(0, 2, "DiracNet")
sheet1.write(0, 3, "&-DiracNet")
for i in range(len(trust)):
    sheet1.write(i + 1, 0, (trust[i]))
    sheet1.write(i + 1, 1, (resnet[i]))
    sheet1.write(i + 1, 2, (diracnet[i]))
    sheet1.write(i + 1, 3, (new_diracnet[i]))
f1.save(xlsx_dir + "3 net predict2.xlsx")
#plt.plot(trust,'r-')
#plt.plot(resnet,'k--')
#plt.plot(diracnet,'k-.')
#plt.plot(new_diracnet,'b-')
#plt.title("Comparison of predicted values of three networks and true values")
#plt.legend(["Trust","ResNet50","DiracNet12","&-DiracNet12"])
#plt.legend(["real visibility","predict visibility"])
#plt.grid()
#plt.savefig('./pppp_predict.png')
#plt.show()