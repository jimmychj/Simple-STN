import sys, os
from neuron import h
import neuron as nrn
from neuron.units import ms, mV
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd()
nrn.load_mechanisms(path+'/sth')
h.load_file('stdrun.hoc')
PI = np.pi
fiberD=2
paralength1=3  
nodelength=1
space_p1=0.002  
space_p2=0.004
space_i=0.004
rhoa = 7e5
nodeD = 1.4
axonD = 1.6
paraD1 = 1.4
paraD2 = 1.6
mycm=0.1
mygm = 0.001
nl=30
Rpn0=(rhoa*.01)/(PI*((((nodeD/2)+space_p1)**2)-((nodeD/2)**2)))
Rpn1=(rhoa*.01)/(PI*((((paraD1/2)+space_p1)**2)-((paraD1/2)**2)))
Rpn2=(rhoa*.01)/(PI*((((paraD2/2)+space_p2)**2)-((paraD2/2)**2)))
Rpx=(rhoa*.01)/(PI*((((axonD/2)+space_i)**2)-((axonD/2)**2)))


def read_dat(file_name):
    df = pd.read_csv('sth/sth-data/' + file_name, header=None, sep=' ')
    df = df.sort_values(by=[0])
    df = df.reset_index(drop=True)
    return df


def read_channel_distribution(g_name, c_name):
    df = pd.read_csv('sth/sth-data/cell-'+g_name+'_'+c_name, header=None, sep=' ')
    df.columns = ['sec_name', 'sec_ref', 'seg', 'val']
    return df


def find_dat(i, df):
    children = [df[1][i]-1, df[2][i]-1]
    diam = df[3][i]
    L = df[4][i]
    nseg = df[5][i]
    return [children, diam, L, nseg]


def insert_channels(sec):
    sec.insert('STh')
    sec.insert('Na')
    sec.insert('NaL')
    sec.insert('KDR')
    sec.insert('Kv31')
    sec.insert('Ih')
    sec.insert('Cacum')
    sec.insert('sKCa')
    sec.insert('CaT')
    sec.insert('HVA')
    sec.insert('extracellular')
    for i in range(1):
        sec.xraxial[i] = 1e9
        sec.xg[i] = 1e9
        sec.xc[i] = 0


def set_aCSF(i):
    if i == 4:
        # print("Setting in vitro parameters based on Atherton (2010)")
        h.nai0_na_ion = 18
        h.nao0_na_ion = 153.25
        h.ki0_k_ion = 168
        h.ko0_k_ion = 3   
        h.cai0_ca_ion = 8e-05
        h.cao0_ca_ion = 1.6    
#         h.cli0_cl_ion = 4
#         h.clo0_cl_ion = 135
    if i == 3:
        # print("Setting in vitro parameters based on Bevan & Wilson (1999)")
        h.nai0_na_ion = 15             # This is different in Miocinovic model
        h.nao0_na_ion = 128.5          # This is different in Miocinovic model
        h.ki0_k_ion = 140
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.0
#         h.cli0_cl_ion = 4
#         h.clo0_cl_ion = 132.5
    if i == 0:
        # print("WARNING: Using NEURON defaults for in vitro parameters")
        h.nai0_na_ion = 10
        h.nao0_na_ion = 140
        h.ki0_k_ion = 54
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 5e-05
        h.cao0_ca_ion = 2
#         h.cli0_cl_ion = 0
#         h.clo0_cl_ion = 0


# Building Cell
class CreateSth():
    def __init__(self, params=None):
        self.s = params
        self.rhoa= 7e5
        self.soma = h.Section(name='soma', cell=self)
        self.initseg = h.Section(name='initseg', cell=self)
        self._setup_morphology()
        self.sdi = [self.soma, self.dend0, self.dend1, self.initseg]
        self.all = self.sdi

        self.r_kdr = h.Vector(1)
        self.r_kdr.x[0] = self.s[25]
        self.r_na = h.Vector(1)
        self.r_na.x[0] = self.s[26]
        self.r_kv31 = h.Vector(1)
        self.r_kv31.x[0] = self.s[27]
        self.r_cat = h.Vector(1)
        self.r_cat.x[0] = self.s[28]
        self.v_kdr = h.Vector(1)
        self.v_kdr.x[0] = 0
        self.v_kv31 = h.Vector(1)
        self.v_kv31.x[0] = 0
        self.r_na_inact = h.Vector(1)
        self.r_na_inact.x[0] = self.s[26]
        self._setup_biophysics()
        
    def _setup_morphology(self):
        self.soma.diam = 18.3112
        self.soma.L = 18.8
        self.initseg.diam = 1.8904976874853334
        self.initseg.L = 21.7413353424173
        self.dend1 = []
        self.dend0 = []
        
        # Building Tree 0
        self.dend0 = h.Section(name='dend0', cell=self)
        self.dend0.diam = 2
        self.dend0.L = 200
        self.dend0.nseg = 1
        
        # Building Tree 0
        self.dend1 = h.Section(name='dend1', cell=self)
        self.dend1.diam = 2
        self.dend1.L = 200
        self.dend1.nseg = 1

        # Connecting Dend0 and Dend1
        self.dend0.connect(self.soma, 0)
        self.dend1.connect(self.dend0)
        # Connecting axon and soma
        self.initseg.connect(self.soma, 1)


    def _setup_biophysics(self):
        for sec in self.sdi:
            sec.Ra = 150
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
            insert_channels(sec)

        for seg in self.soma:
            seg.HVA.gcaL = self.s[1]
            seg.HVA.gcaN = self.s[0]
            seg.CaT.gcaT = 0
            seg.Ih.gk = self.s[2]
            seg.KDR.gk = self.s[3]
            seg.Kv31.gk = self.s[4]
            seg.sKCa.gk = self.s[5]
            seg.NaL.gna = self.s[6]
            seg.Na.gna = self.s[7]
            seg.STh.gpas = self.s[8]
            h.setpointer(self.r_kdr._ref_x[0], 'r_kdr', seg.KDR)
            h.setpointer(self.v_kdr._ref_x[0], 'v_kdr', seg.KDR)
            h.setpointer(self.r_na._ref_x[0], 'r_na', seg.Na)
            h.setpointer(self.r_na_inact._ref_x[0], 'r_na_inact', seg.Na)
            h.setpointer(self.r_kv31._ref_x[0], 'r_kv31', seg.Kv31)
            h.setpointer(self.v_kv31._ref_x[0], 'v_kv31', seg.Kv31)
            h.setpointer(self.r_cat._ref_x[0], 'r_cat', seg.CaT)
            seg.Cacum.buftau = 50 * self.s[29]
            
        for seg in self.initseg:
            seg.HVA.gcaL = 0
            seg.HVA.gcaN = 0
            seg.CaT.gcaT = 0
            seg.Ih.gk = 0
            seg.KDR.gk = self.s[3] * 15
            seg.Kv31.gk = self.s[4] * 15
            seg.sKCa.gk = 0
            seg.NaL.gna = self.s[6] * 15
            seg.Na.gna = self.s[7] * 15
            seg.STh.gpas = self.s[8]
            h.setpointer(self.r_kdr._ref_x[0], 'r_kdr', seg.KDR)
            h.setpointer(self.v_kdr._ref_x[0], 'v_kdr', seg.KDR)
            h.setpointer(self.r_na._ref_x[0], 'r_na', seg.Na)
            h.setpointer(self.r_na_inact._ref_x[0], 'r_na_inact', seg.Na)
            h.setpointer(self.r_kv31._ref_x[0], 'r_kv31', seg.Kv31)
            h.setpointer(self.v_kv31._ref_x[0], 'v_kv31', seg.Kv31)
            h.setpointer(self.r_cat._ref_x[0], 'r_cat', seg.CaT)
            seg.Cacum.buftau = 50 * self.s[29]

        for seg in self.dend0:
            seg.HVA.gcaL = self.s[16]
            seg.HVA.gcaN = self.s[15]
            seg.CaT.gcaT = self.s[9]
            seg.Ih.gk = self.s[2]
            seg.KDR.gk = self.s[10]
            seg.Kv31.gk = self.s[11]
            seg.sKCa.gk = self.s[12]
            seg.NaL.gna = self.s[13]
            seg.Na.gna = self.s[14]
            seg.STh.gpas = self.s[8]
            h.setpointer(self.r_kdr._ref_x[0], 'r_kdr', seg.KDR)
            h.setpointer(self.v_kdr._ref_x[0], 'v_kdr', seg.KDR)
            h.setpointer(self.r_na._ref_x[0], 'r_na', seg.Na)
            h.setpointer(self.r_na_inact._ref_x[0], 'r_na_inact', seg.Na)
            h.setpointer(self.r_kv31._ref_x[0], 'r_kv31', seg.Kv31)
            h.setpointer(self.v_kv31._ref_x[0], 'v_kv31', seg.Kv31)
            h.setpointer(self.r_cat._ref_x[0], 'r_cat', seg.CaT)
            seg.Cacum.buftau = 30 * self.s[30]

        for seg in self.dend1:
            seg.HVA.gcaL = self.s[24]
            seg.HVA.gcaN = self.s[23]
            seg.CaT.gcaT = self.s[17]
            seg.Ih.gk = self.s[2]
            seg.KDR.gk = self.s[18]
            seg.Kv31.gk = self.s[19]
            seg.sKCa.gk = self.s[20]
            seg.NaL.gna = self.s[21]
            seg.Na.gna = self.s[22]
            seg.STh.gpas = self.s[8]
            h.setpointer(self.r_kdr._ref_x[0], 'r_kdr', seg.KDR)
            h.setpointer(self.v_kdr._ref_x[0], 'v_kdr', seg.KDR)
            h.setpointer(self.r_na._ref_x[0], 'r_na', seg.Na)
            h.setpointer(self.r_na_inact._ref_x[0], 'r_na_inact', seg.Na)
            h.setpointer(self.r_kv31._ref_x[0], 'r_kv31', seg.Kv31)
            h.setpointer(self.v_kv31._ref_x[0], 'v_kv31', seg.Kv31)
            h.setpointer(self.r_cat._ref_x[0], 'r_cat', seg.CaT)
            seg.Cacum.buftau = 30 * self.s[30]
            
        for sec in self.sdi:
            h.ion_style("na_ion",1,2,1,0,1, sec=sec)
            h.ion_style("k_ion",1,2,1,0,1, sec=sec)
            h.ion_style("ca_ion",3,2,1,1,1, sec=sec)

    def __repr__(self):
        return 'sth'

