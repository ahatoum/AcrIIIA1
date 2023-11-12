import pysam
import pickle
import time
import unittest
import textwrap

import matplotlib.pyplot as plt
import matplotlib.colors

import numpy as np
import pandas as pd
import os
import re
import operator
import subprocess

from statistics import median, mean, stdev
from functools import reduce
from pandas.testing import assert_frame_equal
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord  # from Bio.Seq import Seq
from collections import OrderedDict, Counter

from sjsMapper import IGV_Mapper, dvarGen

from mapping_position import Mapping2

from main_module import read_file, write_file, get_kmers, run_clustalo, copy_file, ensure_directory
from gbkGenome import gbkGenome

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", 6000, "display.max_columns", 6000, "display.max_colwidth", 50)

codon_dic = {

        "GCG": "Ala",
        "GCA": "Ala",
        "GCT": "Ala",
        "GCC": "Ala",
        "TGT": "Cys",
        "TGC": "Cys",
        "GAT": "Asp",
        "GAC": "Asp",
        "GAG": "Glu",
        "GAA": "Glu",
        "TTT": "Phe",
        "TTC": "Phe",
        "GGG": "Gly",
        "GGA": "Gly",
        "GGT": "Gly",
        "GGC": "Gly",
        "CAT": "His",
        "CAC": "His",
        "ATA": "Ile",
        "ATT": "Ile",
        "ATC": "Ile",
        "AAG": "Lys",
        "AAA": "Lys",
        "TTG": "Leu",
        "TTA": "Leu",
        "CTG": "Leu",
        "CTA": "Leu",
        "CTT": "Leu",
        "CTC": "Leu",
        "ATG": "Met",
        "AAT": "Asn",
        "AAC": "Asn",
        "CCG": "Pro",
        "CCA": "Pro",
        "CCT": "Pro",
        "CCC": "Pro",
        "CAG": "Gln",
        "CAA": "Gln",
        "AGG": "Arg",
        "AGA": "Arg",
        "CGG": "Arg",
        "CGA": "Arg",
        "CGT": "Arg",
        "CGC": "Arg",
        "AGT": "Ser",
        "AGC": "Ser",
        "TCG": "Ser",
        "TCA": "Ser",
        "TCT": "Ser",
        "TCC": "Ser",
        "ACG": "Thr",
        "ACA": "Thr",
        "ACT": "Thr",
        "ACC": "Thr",
        "GTG": "Val",
        "GTA": "Val",
        "GTT": "Val",
        "GTC": "Val",
        "TGG": "Trp",
        "TAT": "Tyr",
        "TAC": "Tyr",
        "TGA": "End",
        "TAG": "End",
        "TAA": "End"}

class Utilities:

    @staticmethod
    def print_colored(text, color):

        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'black': '\033[98m',
            'end': '\033[0m',
        }
        print(colors[color] + text + colors['end'])

    @staticmethod
    def get_sub_tnumreads(path_sortd):

        samfile = pysam.AlignmentFile(path_sortd, "rb")
        out = samfile.get_index_statistics()
        total_mapped = out[0].mapped
        contig = out[0].contig
        it = list(samfile.fetch(contig, 2517620, 2517868))
        sub_mapped = len([read for read in it if not read.is_unmapped])
        samfile.close()

        print(f"total_mapped: {total_mapped}")
        print('sub_mapped: ', sub_mapped)

        return total_mapped - sub_mapped

    @staticmethod
    def auxiliary_printer_for_genbank_file(path):

        record: SeqRecord = SeqIO.read(path, "gb")
        features = [feature for feature in record.features if feature.type != "gene" and feature.type != "source"]
        print(len(features))
        for f in features[0:5]: print(f, '\n')

class ParseSS:

    class UnitSS:

        def __init__(self, name_line, type_line, seq_line, str_line):

            self._data_dict = OrderedDict()

            self['Name'] = name_line
            self['Type'] = type_line
            self['Seq'] = seq_line
            self['Str'] = str_line

            self["Str_Parsed"] = self._parse_str()
            self["Type_Parsed"] = self._parse_type()

            for k, v in self["Str_Parsed"].items():

                if (v['seq'] == self["Type_Parsed"]["isotype"]) and (v['mp'] == self["Type_Parsed"]["start"]): v[
                    'anticodon'] = True

        def __str__(self):

            s = ''

            s += f"\t{'Name'.ljust(6, ' ')}{self['Name']}\n"
            s += f"\t{'Type'.ljust(6, ' ')}{self['Type']}\n"
            s += f"\t{'Seq'.ljust(6, ' ')}{self['Seq']}\n"
            s += f"\t{'Str'.ljust(6, ' ')}{self['Str']}\n"

            s += "\n\tStr_Parsed\n\n"
            for k, v in self['Str_Parsed'].items():
                s += f"\t\t{k}\t{v}\n"

            s += "\n\tType_Parsed\n\n"
            for k, v in self['Type_Parsed'].items():
                s += f"\t\t{k}\t{v}\n"

            return s

        def __setitem__(self, key, value):

            self._data_dict[key] = value

        def __getitem__(self, item):

            return self._data_dict[item]

        def _parse_str(self):

            loops_dict = dict()

            i = 0

            for match in re.finditer(r'>\.{3,25}<', self['Str']):
                a, b = match.span()

                st = a + 1
                ed = b - 1

                loop_length = b - a - 2
                loop_is_odd = loop_length % 2
                loop_is_valid = (loop_length >= 4) and (loop_is_odd == 1)
                mid_point = int(st + (ed - 1 - st) / 2)  # mid_point = int(a + (b - 1 - a) / 2) # original

                res = self['Seq'][mid_point - 1: mid_point + 2]

                loops_dict[i] = {'seq': res,
                                 'mp': mid_point,
                                 'start': st,
                                 'end': ed,
                                 'odd': loop_is_odd,
                                 'valid': loop_is_valid,
                                 'anticodon': False}
                i += 1

            return loops_dict

        def _parse_type(self):

            m = re.search(
                r'^([a-zA-Z]{2,5}[0-9]*)\s+Anticodon:\s+([A-Z]{3})\s+at\s+([0-9]{1,2})\s*-\s*([0-9]{1,2}).+Score:\s+(.+)$',
                self['Type'])

            trna_type = m.group(1)
            isotype = m.group(2)
            start = m.group(3)
            stop = m.group(4)
            score = m.group(5)

            vals = OrderedDict()

            vals["trna_type"] = str(trna_type)
            vals["isotype"] = str(isotype)
            vals["start"] = int(start)
            vals["stop"] = int(stop)
            vals["score"] = float(score)

            return vals

    def __init__(self, path, force = False):

        self.path = path
        self.path_save = os.path.dirname(path) + '/parsed.ss'
        self._data_dict = OrderedDict()

        if os.path.isfile(self.path_save) and not force:

            print('<<<<<<<<<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('        The parsed.ss already exists. Reading.       ')
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>>>>>>>>>')
            print()

            self._data_dict = self.load()

        else:

            lines = read_file(self.path, mode = 'lines')

            str_locs = list()

            for i, line in enumerate(lines):

                if line.startswith('Str:'): str_locs.append(i)

            for str_loc in str_locs:

                name_line =  lines[str_loc - 4]
                type_line = lines[str_loc - 3][6:]
                seq_line  = lines[str_loc - 1][5:]
                str_line  = lines[str_loc][5:]

                self[name_line.split('.trna')[0]] = self.UnitSS(name_line, type_line, seq_line, str_line)

                self.pickle()

    def __str__(self):

        s = ''

        for name in self:

            s += '\n' + 50 * '-' + '\n'
            s += f"\n{name}\n"
            s += str(self[name])

        return s

    def __setitem__(self, key, value):

        self._data_dict[key] = value

    def __getitem__(self, item):

        return self._data_dict[item]

    def __iter__(self):

        return iter(self._data_dict)

    def __len__(self):

        return len(self._data_dict)

    def pickle(self):

        with open(self.path_save, 'wb') as f: pickle.dump(self._data_dict, f)

    def load(self):

        with open(self.path_save, 'rb') as f: _data_dict = pickle.load(f)
        return _data_dict

    def self_test(self):

        print(len(self))
        print(self)

class ParseInfernal:

    def __init__(self, path, delete_list):

        self.cons_line = None
        self.rf_line = None
        self._Alignment_Dict = OrderedDict()
        self.name_len = 0

        lines = read_file(path, mode='lines')

        for i, line in enumerate(lines):

            if line.startswith('tRNA_'):

                name, almt = line.split()

                self._Alignment_Dict[name] = almt.upper().replace('U', 'T')

            elif line.startswith('#=GC SS_cons'): self.cons_line = line.split()[2]

            elif line.startswith('#=GC RF'): self.rf_line = line.split()[2]

            else: pass

        for item in delete_list:

            del self._Alignment_Dict[item]

    def __str__(self):

        nl = self.get_name_len()

        s = ''

        for name in self: s += f"\n{name}: {self[name]}"

        s += f"\n{nl * ' '}: {self.cons_line}"
        s += f"\n{nl * ' '}: {self.rf_line}"

        return s

    def __getitem__(self, item):

        return self._Alignment_Dict[item]

    def __iter__(self):

        return iter(self._Alignment_Dict)

    def __len__(self):

        return len(self._Alignment_Dict)

    def get_name_len(self):

        v = []
        for name in self: v.append(len(name))
        return v[0]

    def verify(self):

        for name in self:

            a = name[26:29]
            b = self[name][36:39]
            print(f"{a} ::: {b} -> {a==b}")

class MapperAN1(IGV_Mapper):

    class Interval:

        def __init__(self, pair):
            self.interval = pair
            self.length = pair[1] - pair[0]
            self.mean_rpm = 0
            self.median_rpm = 0
            self.max_rpm = 0

    class IntervalCollection:

        def __init__(self, DataVec, peak_threshold, zero_threshold):

            print('Analyzing Intervals Now...')

            self.peak_intervals = list()
            self.trth_intervals = list()

            signum = np.nan_to_num(np.diff(np.sign(DataVec - zero_threshold)))
            zero_crossings = np.where(signum != 0)[0]
            pairs = IGV_Mapper.slice_list(zero_crossings, 2)

            for a, b in pairs:

                z = MapperAN1.Interval((a, b))
                z.mean_rpm = np.mean(DataVec[a:b])
                z.median_rpm = np.median(DataVec[a:b])
                z.max_rpm = max(DataVec[a:b])

                if z.mean_rpm > peak_threshold:
                    self.peak_intervals.append(z)
                    print('peak', a, b)
                else:
                    self.trth_intervals.append(z)
                    print('trth', a, b)

        @staticmethod
        def debug():

            pdata = "/media/yourusername/Samsung980PRO/biodata2/sequencing/rna_sequence/NZ_CP083274.1/array_data"
            plocs = "/media/yourusername/Samsung980PRO/biodata2/sequencing/rna_sequence/NZ_CP083274.1/array_locs"

            with open(pdata, 'rb') as f: array_data = pickle.load(f)
            with open(plocs, 'rb') as f: array_locs = pickle.load(f)

            zero_th = 500
            peak_th = 700
            IC = MapperAN1.IntervalCollection(array_data, peak_th, zero_th)

            plt.plot(array_locs, array_data)
            plt.axhline(zero_th, color='red')
            plt.axhline(peak_th, color='yellow')

            for z in IC.peak_intervals:
                a, b = z.interval

                plt.axvline(a, color='black')
                plt.axvline(b, color='orange')

            plt.show()

    def __init__(self, dvar):

        super().__init__(dvar)

        self.x, self.y, self.t, self.RPM, self.Intervals = None, None, None, None, None

    def __str__(self):

        SEPERATOR = '\t' + 50 * '.' + '\n'
        # s = f"{SEPERATOR}\tdvar\n{SEPERATOR}\n"

        s = ''
        s += '\t' + str(self.dvar).replace('\t', '\t\t')

        if hasattr(self, "pss"):

            s += f"\n{SEPERATOR}\tParsed SS -> Self\n{SEPERATOR}\n"
            s += self.read_parsed_ss().__str__()

        return s

    def attach_parsed_ss(self, PSS):

        self.__setattr__("pss", PSS)

    def read_parsed_ss(self):

        PSS: ParseSS = self.__getattribute__("pss")
        val_dict = PSS[self.name] # type is ParseSS.UnitSS

        return val_dict

    def analyze_start_ends_with_peak_detection(self, path_figures, peak_threshold=1000, zero_threshold=500):

        ll=0 # x_left=7424, x_right=7689

        x, y, t, RPM = self.extract_xyt_and_rpm()
        s, e = self._generate_binned_data(x)
        print('M stats:', len(x), len(s), len(t))

        IC = self.IntervalCollection(t, peak_threshold, zero_threshold) # (stb, edb, x[:-1], t) # x0, x1 = 7424, 7689

        for i, z in enumerate(IC.peak_intervals):

            a, b = z.interval

            sn = s[a-ll:b+ll]
            en = e[a-ll:b+ll]
            tn = t[a-ll:b+ll]
            xn = x[a-ll:b+ll]

            cx = np.cumsum(sn-en)

            print('plotting interval {}, {}'.format(a,b))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4.5), dpi=300)

            xlocal = xn-min(xn)
            ax.bar(xlocal, sn , width = 0.5)
            ax.bar(xlocal, -en, width = 0.5)
            ax.bar(xlocal, tn , width = 0.5, alpha=0.5, color='black')

            ax.plot(xlocal, cx)
            ax.grid(which='major', alpha=0.7)

            ax.set_ylabel('Coverage (Raw)')
            ax.set_title(f"{a - ll} - {b + ll}")
            ax.set_axisbelow(True)

            plt_name = '{}_{}_zoom_start_end.png'.format(a, b)
            print(path_figures + plt_name)
            plt.savefig(path_figures + plt_name) # plt.show()
            plt.close()

    def analyze_start_ends_without_peak_detection(self, path_figures, path=''):

        up_color = (50/255, 200/255, 50/255)
        dw_color = 'red'
        color_coverage_bars = "black"
        color_coverage_alpha = 0.6
        color_loops = "gray"

        x, y, t, RPM = self.extract_xyt_and_rpm()
        s, e = self._generate_binned_data(x)
        bases = list(self.query_seq)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4.5), dpi=300)
        ax.set_zorder(1)

        if len(bases) > 120:

            print('\n\tLength of sequence too long to display letters...')

            ax.fill_between(x, y1=  s, color=up_color, linewidth=0.5, zorder=2)
            ax.fill_between(x, y1= -e, color=dw_color, linewidth=0.5, zorder=2)
            ax.fill_between(x, y1=  t, alpha=color_coverage_alpha, facecolor=color_coverage_bars, edgecolor='none')

        else:

            ax.bar(x, s  , width=0.5, color = up_color, zorder=2)
            ax.bar(x, -e , width=0.5, color = dw_color, zorder=2)
            ax.bar(x, t  , width=0.5, alpha=color_coverage_alpha, color=color_coverage_bars)

            if hasattr(self, "pss"):

                val_dict = self.read_parsed_ss()
                for k, d in val_dict['Str_Parsed'].items():

                    ax.axvspan(d['start'], d['end'] - 1, zorder=0, alpha=0.6 if d['anticodon'] else 0.2, edgecolor='none', facecolor=color_loops)

                marks = list(val_dict['Str'])
                annot = [f"{mark}\n{base}" for base, mark in zip(bases, marks)]

            else:

                annot = bases

            x_ticks = list(annot)
            ax.set_xticks(x)
            ax.set_xticklabels(x_ticks, fontsize=6)
            ax.xaxis.label.set_color('green')

        # Final Settings

        ax.set_ylabel('Coverage (Raw)')
        ax.set_title(self.name)
        ax.set_axisbelow(True)

        plt_name = f'{self.name}_start_end.png'

        if path != '': path_save = path + plt_name
        else:          path_save = path_figures + plt_name

        plt.savefig(path_save)  # plt.show()
        plt.close()

    def plot_whole_with_scipy_peaks(self, mode='RPM', show_inset = False, prominence = 200, path_gb='', path=''):

        import matplotlib.ticker as ticker
        from scipy.signal import find_peaks

        def format_axis(ax, fsize, labelsize, xlabel='', ylabel=''):

            ax.set_yscale('log')

            if xlabel != '':

                ax.set_xlabel(xlabel, fontsize=fsize)

            if ylabel != '':

                ax.set_ylabel(ylabel, fontsize=fsize)

            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', which='major', labelsize=labelsize)
            ax.set_ylim([10, 250000])
            # ax.set_xlim([1450000, 1452500])

        def export_annotated_peaks(positions, peaks, path_gb):

            info = self.get_position_annotations(path_gb, positions)

            rows = list()

            for position, peak in zip(positions, peaks):

                # print(f"pos: {position}, peak: {peak}, info: {info[position]}")

                rows.append([position, peak, *info[position]])

            df = pd.DataFrame(rows, columns = ['Pos', 'Peak_Cov', 'Locus Tag', 'Product'])

            df.set_index('Pos', inplace=True)

            df = df.sort_values(by='Peak_Cov', ascending=False)

            return df

        if path != '':

            path_csv = path + f"{self.name}_whole_annotated_peaks.csv"
            path_png = path + f"{self.name}_whole_annotated_peaks.png"

        else:

            path_csv = self.root + f"{self.name}_whole_annotated_peaks.csv"
            path_png = self.root + f"{self.name}_whole_annotated_peaks.png"

        lw = 0.3
        fsize = 10

        x, y, t, RPM = self.extract_xyt_and_rpm()

        RPM = np.nan_to_num(RPM)
        t = np.nan_to_num(t)

        print(len(x), len(RPM))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4.5), dpi=300)

        if mode == 'RPM': data = [RPM, "Depth of Coverage (RPM)"]

        else: data = [t, "Raw Coverage"]

        # --------------------------------------------------

        peaks, _ = find_peaks(data[0], prominence=prominence)

        if path_gb != '':

            df = export_annotated_peaks(x[peaks], data[0][peaks], path_gb)
            df.to_csv(path_csv)
            print(df)

        ax.scatter(x[peaks], data[0][peaks], s=0.2)

        ax.plot(x, data[0], color='black', linewidth=lw)
        format_axis(ax, fsize=fsize, labelsize=8, xlabel = 'Genome Position', ylabel = data[1])

        # ax.axhline(y=100, alpha=0.3, color='grey')

        ax.fill_between(x, y1=5e5, y2=10000, alpha=0.2, color='yellow')

        if show_inset:

            ax2 = fig.add_axes([0.30, 0.70, 0.20, 0.15])
            ax2.plot(x, data[0], color='black', linewidth=lw)
            format_axis(ax2, fsize, labelsize=5)
            ax2.set_xlim([2e5, 2.25e5])
            ax2.set_ylim([0, 200])

        fig.suptitle(self.nm2, fontsize=fsize)
        plt.tight_layout()
        plt.savefig(path_png); plt.close()

        # plt.show()

    def pysam_exp(self):

        def print_bam_info():
            samfile: pysam.libcalignmentfile.AlignmentFile = pysam.AlignmentFile(self.path_sortd, "rb")

            print(samfile.get_index_statistics())
            print()
            print(samfile.description)
            print()
            print(samfile.header)
            print()
            print(samfile.references)

        def stats():

            samfile = pysam.AlignmentFile(self.path_sortd, "rb")

            out = samfile.get_index_statistics()

            contig = out[0].contig
            total = out[0].total
            mapped = out[0].mapped
            unmapped = out[0].unmapped

            samfile.close()

            print(contig)
            print(total)
            print('2 x 2,298,519: ',2*2298519, f'68.61%: {0.6861*2*2298519}')
            print(f"mapped: {mapped}, unmapped: {unmapped}")
            print(mapped + unmapped)

        def unmapped():

            samfile = pysam.AlignmentFile(self.path_sortd, "rb")
            read: pysam.libcalignedsegment.AlignedSegment

            i = 0
            for read in samfile:

                if read.is_unmapped:
                    print(read)
                    i += 1
                    if i > 5: break

            print('\n\n')

            i = 0
            for read in samfile:

                if not read.is_unmapped:
                    print(read)
                    i += 1
                    if i > 5: break

        samfile = pysam.AlignmentFile(self.path_sortd, "rb")
        out = samfile.get_index_statistics()
        total_mapped = out[0].mapped

        it = list(samfile.fetch('gi|57865352|ref|NC_002976.3|', 2517620, 2517868))

        print(f"total_mapped: {total_mapped}")

        sub_mapped = len([read for read in it if not read.is_unmapped])
        sub_all    = len(list(it))

        print('sub_mapped: ',sub_mapped)
        print('sub_all: ', sub_all)
        samfile.close()

    # -----

    def _generate_binned_data(self, x):

        samfile: pysam.libcalignmentfile.AlignmentFile = pysam.AlignmentFile(self.path_sortd, "rb")
        read: pysam.libcalignedsegment.AlignedSegment
        print("\n\tGenerating Binned Data using Pysam. Stats are: ", samfile.get_index_statistics())

        start_coordinates = list()
        end_coordinates = list()

        for read in samfile:

            if (not read.is_unmapped) and (read.mapping_quality >= 0):

                start_coordinates.append(read.reference_start)  # todo: add one to start
                end_coordinates.append(read.reference_end-1)

        samfile.close()

        start_coordinates = np.array(start_coordinates)
        end_coordinates = np.array(end_coordinates)

        x = np.append(x, max(x)+1)

        stb, _ = np.histogram(start_coordinates, bins=x)
        edb, _ = np.histogram(end_coordinates, bins=x)

        # for i, stp in enumerate(st): print(i, bins[i], st[i], ed[i])

        return stb, edb

    @staticmethod
    def find_read_by_name(path, read_name):

        # read_name = "M03023:712:000000000-DGFRK:1:1101:16205:9325"

        samfile: pysam.libcalignmentfile.AlignmentFile = pysam.AlignmentFile(path, "rb")

        index = pysam.IndexedReads(samfile)
        index.build()

        iterator = index.find(read_name)
        read: pysam.libcalignedsegment.AlignedSegment
        for read in iterator:
            print(read)
            print(read.reference_start, read.reference_end)
            print('\n\n')

    @staticmethod
    def analyze_whole_annotated_peaks_csv(path_whole_annotated_peaks):

        ltag_dict = dict()

        df1 = pd.read_csv(path_whole_annotated_peaks)

        for ind in df1.index:

            ltag = df1.loc[ind]['Locus Tag']
            prod = df1.loc[ind]['Product']
            ltag_dict[ltag] = prod

        print('\n' + 50 * '#' + '\n')

        rrr = len(set(df1['Locus Tag'].to_list()))
        n_keys = len(ltag_dict.keys())

        print(rrr, n_keys)
        for k, v in ltag_dict.items(): print(k, v)

    @staticmethod
    def get_position_annotations(path_gb, positions):

        record = SeqIO.read(path_gb, "gb")
        info = {pos: (None, None) for pos in range(len(record.seq))}

        features = [f for f in record.features if f.type not in {"gene", "source"}]

        for feature in features:

            quals = feature.qualifiers
            ltag = quals.get('locus_tag', ['None'])[0]
            prod = quals.get('product', ['None'])[0]

            for pos in feature.location: info[pos] = (ltag, prod)

        return {int(pos): info[int(pos)] for pos in positions}

class ProcessorAN1:

    @staticmethod
    def main():

        u = 3 #10
        j = 1 #10

        path_root    = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/"
        path_ss_file = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/NZ_CP083274_tRNAscan_SE_Results/seq84479.ss"
        path_gb      = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/NZ_CP083274.1.gb"

        # Batch dvar processing
        if u == 0:

            pathq = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/all_queries/"
            dvars = dvarGen(path_root + "mappings_3/", path_queries=pathq)

            # Classified :: analyze_start_ends_without_peak_detection()
            if j == 0:

                PSS = ParseSS(path_ss_file)

                for name in dvars:

                    IGVM = MapperAN1(dvars[name])
                    IGVM.printing_enabled = True
                    print(IGVM)
                    IGVM.execute_all()
                    IGVM.calculate_coverage()

                    if name == "NZ_CP083274":

                        path = dvars.path_root + "figures/" + "whole_analysis/"
                        IGVM.plot_whole_with_scipy_peaks(mode='Raw',
                                                         show_inset=False,
                                                         prominence=200,
                                                         path_gb=path_gb,
                                                         path=path)

                    elif name.startswith('tRNA_'):

                        IGVM.attach_parsed_ss(PSS)
                        IGVM.analyze_start_ends_without_peak_detection(dvars.path_root + "figures/" + "tRNA/")

                    else:

                        IGVM.analyze_start_ends_without_peak_detection(dvars.path_root + "figures/" + "other/")

            # Mapping individuals by name
            if j == 4:

                name = "16S_ribosomal_RNA_K6U01_RS21610"
                # name = "transfer_messenger_RNA_K6U01_RS19150"

                IGVM = MapperAN1(dvars[name])
                IGVM.printing_enabled = True
                print(IGVM)
                print("Total Number of Mapped Reads: ", IGVM.read_total_number_of_read_for_rpm())
                print(IGVM.path_tnumreads)

            # Whole plot for NZ_CP083274
            if j == 5:

                IGVM = MapperAN1(dvars["NZ_CP083274"])
                IGVM.printing_enabled = True
                print(IGVM)
                IGVM.execute_all()
                IGVM.calculate_coverage()

                path = dvars.path_root + "figures/" + "whole_analysis/"
                IGVM.plot_whole_with_scipy_peaks(mode='Raw',
                                                 show_inset=False,
                                                 prominence=200,
                                                 path_gb=path_gb,
                                                 path=path)

            # tRNA_summary_plotter
            if j == 6:

                TP = OlderClasses.tRNA_summary_plotter(dvars, path_ss_file)
                TP.plot()
                # df = TP.summary_to_dataframe()

        # Individual dvar processing
        if u == 1:

            dvars = dvarGen(path_root + "mappings_fake/")
            path_figures = dvars.path_root + "figures/"

            IGVM = MapperAN1(dvars['tRNA_Gln_K6U01_RS16055'])

            # bowtie2 Map and calculate coverage
            if j == 1:

                IGVM.printing_enabled = True
                IGVM.execute_all()
                IGVM.calculate_coverage()
                IGVM.plot_whole_with_scipy_peaks(mode='Raw')

            # plot coverage peaks
            if j == 2:

                st = time.perf_counter()
                IGVM.analyze_start_ends_with_peak_detection(path_figures)
                ed = time.perf_counter()
                print('Finished in:', ed - st)

            # Whole Plots
            if j == 3:

                IGVM.plot_whole_with_scipy_peaks(mode='Raw')
                pass

            # debug: print Mapping2.data_frame
            if j == 4:

                M: Mapping2 = IGVM.ensure_parsed_wig()
                print(M.data_frame)

            # analyze start-ends
            if j == 5:

                print("Read from file - Total Number reads: ", IGVM.read_total_number_of_read_for_rpm())

                # IGVM.analyze_start_ends_with_peak_detection(peak_threshold=15000, zero_threshold=5000) # 150, 95
                IGVM.analyze_start_ends_without_peak_detection(path_figures)  # 150, 95

        # Exporters
        if u == 2:

            # Export using vec and export2fasta_by_locus_tag_NUCLEOTIDE
            if j == 0:

                # returns a list of locus_tags of unique non-tRNAs from_all_peaks_merged.csv
                def asdf():
                    pth = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/all_peaks_merged.csv"

                    df = pd.read_csv(pth)

                    df['Merged'] = df['Locus Tag'] + "@" + df['Product']

                    seq_list = [el for el in df['Merged'].to_list() if isinstance(el, str)]

                    seq_set = set(seq_list)

                    seq_list = list(seq_set)

                    print(len(seq_list), len(seq_set))

                    # seq_list = [a.split('@')[0] for a in seq_list if not a.startswith("tRNA-")]
                    seq_list = [a for a in seq_list if not a.split('@')[1].startswith("tRNA-")]
                    seq_list = [a.split('@')[0] for a in seq_list]

                    for a in seq_list: print(a)

                    return seq_list

                path_genbank = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/"  # mappings/queries/"
                path_export  = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/fasta_exports_from_NZ_CP083274/"

                # tRNAs
                vec0 =  ["K6U01_RS00560",
                        "K6U01_RS09930",
                        "K6U01_RS09935",
                        "K6U01_RS09940",
                        "K6U01_RS10110",
                        "K6U01_RS13645",
                        "K6U01_RS15165",
                        "K6U01_RS15175",
                        "K6U01_RS16055",
                        "K6U01_RS16135",
                        "K6U01_RS21000",
                        "K6U01_RS21540",
                        "K6U01_RS21740",
                        "K6U01_RS21860",
                        "K6U01_RS21875",
                        "K6U01_RS21925",
                        "K6U01_RS19150"]

                # <n>S ribosomal RNA and gene=ffs (last one)
                vec1 = ['K6U01_RS06930',
                        'K6U01_RS10120',
                        'K6U01_RS10590',
                        'K6U01_RS10600',
                        'K6U01_RS21075',
                        'K6U01_RS21435',
                        'K6U01_RS21495',
                        'K6U01_RS21610',
                        'K6U01_RS13420']

                # gene ffs -> with padding=30
                vec2 = ['K6U01_RS13420']

                # unique non-tRNAs from_all_peaks_merged.csv
                vec3 = asdf()

                gbk = gbkGenome(path_genbank, 'NZ_CP083274.1')
                gbk.export2fasta_by_locus_tag_NUCLEOTIDE(vec3, path_export)

            # Export using export2fasta_by_keyword_NUCLEOTIDE
            if j == 1:

                path_genbank = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/mappings/queries/"
                path_fasta   = "/media/yourusername/Samsung980PRO/biodata2/anticrispr/exported_trnas/export.fasta"

                gbk = gbkGenome(path_genbank, 'NZ_CP083274.1')
                gbk.export2fasta_by_keyword_NUCLEOTIDE(path_fasta, mode='individual')

        if u == 3:

            pvar0 = {"mapping": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/mappings_1/",
                     "genbank": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/NZ_CP083274.1.gb",
                     "trnaall": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/NZ_CP083274_tRNAscan_SE_Results/input_file.fasta",
                     "uniqfst": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/all_unique_trnas.fasta",
                     "queries": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/all_queries/",
                     "pssfile": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/NZ_CP083274_tRNAscan_SE_Results/seq84479.ss",
                     "infernl": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/all_unique_trnas.infernal",
                     "cmffile": "/media/yourusername/Samsung980PRO/biodata2/anticrispr/trna_classifications/bact-num.cm",
                     "plot_nm": "Ecoli Experiment",
                     "min_cov": 2000,
                     "del_ind": [7, 8, 26, 27, 28, 29, 30, 31, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 104, 108, 111, 112],
                     "removel": ['tRNA_Gly_K6U01_RS13050_Gly_GCC_neg',
                                 'tRNA_Ile_K6U01_RS21565_Ile_CAT_neg',
                                 'tRNA_Val_K6U01_RS00560_Val_GAC_pos',
                                 'tRNA_Val_K6U01_RS06935_Val_TAC_pos',
                                 'tRNA_Tyr_K6U01_RS21740_Tyr_GTA_pos',
                                 'tRNA_Leu_K6U01_RS09930_Leu_CAG_neg',
                                 'tRNA_Thr_K6U01_RS21750_Thr_GGT_pos',
                                 'tRNA_Met_K6U01_RS16025_Met_CAT_pos']}

            pvar1 = {"mapping": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_0_MAPPINGS/mappings_acr_plus_phage_rep0/",
                     "genbank": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/genbank_files/LM1680.gb",
                     "trnaall": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_1_exported_trnas/all.fasta",
                     "uniqfst": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_3_uniques_fasta_for_infernal/all_unique_trnas.fasta",
                     "queries": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_1_exported_trnas/queries/",
                     "pssfile": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_2_tRNAscan_SE_results/seq190944.ss",
                     "infernl": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_4_run_infernal/all_unique_trnas.infernal",
                     "cmffile": "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/trna_analysis/step_4_run_infernal/bact-num.cm",
                     "plot_nm": "LM1680 Experiment",
                     "min_cov": 200,
                     "del_ind": [79, 80],
                     "removel": ["tRNA_Arg_SERP_RS07015_Arg_ACG_neg",
                                 "tRNA_Asn_SERP_RS06740_Asn_GTT_neg",
                                 "tRNA_Gln_SERP_RS06945_Gln_TTG_neg",
                                 "tRNA_Gly_SERP_RS13810_Gly_TCC_neg",
                                 "tRNA_Gly_SERP_RS06935_Gly_TCC_neg",
                                 "tRNA_Lys_SERP_RS00915_Lys_TTT_pos",
                                 "tRNA_Phe_SERP_RS06970_Phe_GAA_neg",
                                 "tRNA_Pro_SERP_RS00935_Pro_TGG_pos",
                                 "tRNA_Thr_SERP_RS07040_Thr_TGT_neg",
                                 "tRNA_Val_SERP_RS07045_Val_TAC_neg"]
                     }

            SSD = SunSetAnalysis(pvar1)
            print('hello! uuu')

            SSD.step5_plot_sunset_diagram()

            # PIN = ParseInfernal(path_inf, [])
            # PIN.verify()
            # print(PIN) # print(PIN.cons_line) # print(PIN.rf_line)

class MapperAN2(IGV_Mapper):

    def __init__(self, dvar, path_gb_root):

        super().__init__(dvar)

        self.path_gff2              = path_gb_root + f"{dvar['nm1']}.gff"
        self.path_gb                = path_gb_root + f"{dvar['nm1']}.gb"
        self.path_counts            = self.root + "counts.txt"
        self.path_counts_normalized = self.root + "counts_normalized.csv"
        self.path_counts_log        = self.root + "counts.log"

    def __str__(self):

        s = '\t' + str(self.dvar).replace('\t', '\t\t')

        return s

    def run_featureCounts(self, numThreads=16):

        if not os.path.isfile(self.path_counts) or not os.path.isfile(self.path_counts_log):

            print(f"\tRunning featureCounts on {self.dvar['nm1']}")

            command = f"featureCounts -T {numThreads} -pC --countReadPairs -a {self.path_gff2} -t gene -g ID " \
                      f"-o {self.path_counts} {self.path_sortd} > {self.path_counts_log} 2>&1"

            print('\t' + command, '\n')
            subprocess.check_call(command, shell=True)

        else:

            # Already done
            print('\t' + "featureCounts already run.")

    def parse_counts(self):

        df = pd.read_table(self.path_counts, comment='#')
        df.columns = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length', 'Counts']

        product_dict = self.get_product_dict_from_genbank()
        df['Products'] = df['Geneid'].map(product_dict)

        df = df.reindex(columns=['Geneid', 'Products', 'Chr', 'Start', 'End', 'Strand', 'Length', 'Counts']) # df.to_csv(self.path_counts_normalized, index=False)

        return df

    def get_product_dict_from_genbank(self):

        product_dict = {}
        record = SeqIO.read(self.path_gb, 'genbank')

        for feature in record.features:

            if 'locus_tag' in feature.qualifiers and 'product' in feature.qualifiers:
                locus_tag = feature.qualifiers['locus_tag'][0]
                product = feature.qualifiers['product'][0]
                product_dict[locus_tag] = product

        return product_dict

    def plot_mapped_read_length_histogram(self):

        bamfile: pysam.libcalignmentfile.AlignmentFile = pysam.AlignmentFile(self.path_sortd, "rb")
        read: pysam.libcalignedsegment.AlignedSegment

        n = 0
        counts = 0
        lengths_bucket = list()
        maxn = 1e9

        for read in bamfile:

            n += 1
            if n > maxn: break
            if not read.is_unmapped:
                counts += 1
                lengths_bucket.append(read.query_length)

        print(f"total mapped: {counts}")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

        bins = np.arange(0, 150 + 1)
        hist, bin_edges = np.histogram(lengths_bucket, bins=bins)
        ax.bar(bin_edges[:-1], hist)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(bin_edges, minor=True)
        ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5)

        ax.set_xlim([0, 151])
        ax.set_ylim([0, 25000])
        plt.show()

def df_trna_tmsr_othr_seperator(df):

    mask1_tRNA = (df['Products'].str.startswith('tRNA-')) & (df['Products'].str.len() < 10)
    mask1_tmsr = df['Products'].str.startswith('transfer-messenger RNA')

    df_trna = df[mask1_tRNA]
    df_tmsr = df[mask1_tmsr]
    df_othr = df[~mask1_tRNA & ~mask1_tmsr]

    return df_trna, df_tmsr, df_othr

def generate_dfCombined(data_mat, path_combined, path_gb_root):

    if not os.path.isfile(path_combined):

        column_names = list()

        df_dict   = dict()
        df_list   = list()
        dvar_dict = dict()

        data_indices = data_mat.index
        print(data_indices); print('\n')
        name = "LM1680"

        for n in data_indices:

            dvar_dict[f"p{n}"] = dvarGen(data_mat["acr_plus_phage"][n])
            dvar_dict[f"m{n}"] = dvarGen(data_mat["acr_minus_phage"][n])
            dvar_dict[f"c{n}"] = dvarGen(data_mat["negative_control"][n])

        for k, dvars in dvar_dict.items():

            print(k, dvars[name])

            IGVM = MapperAN2(dvars[name], path_gb_root)
            IGVM.run_featureCounts()
            df = IGVM.parse_counts().drop(['Strand', 'Chr'], axis=1)
            df.rename(columns={'Counts': k}, inplace=True)

            df_dict[k] = df
            df_list.append(df)
            column_names.append(k)

        on_list = ['Geneid', 'Products', 'Start', 'End', 'Length']
        df_combined = reduce(lambda left, right: pd.merge(left, right, on=on_list, how='outer'), df_list)

        df_combined.to_csv(path_combined)

    else:

        print("df_combined already exists!")

class DEseq2:

    def __init__(self, analysis_name, path_deseq, path_combined, plist, mlist, clist, epsilon='10th'):

        self.analysis_name = analysis_name
        self.path_analysis_root = path_deseq + f"{analysis_name}/"
        ensure_directory(self.path_analysis_root)

        self.epsilon = epsilon
        self.plist = plist
        self.mlist = mlist
        self.clist = clist

        self.path_combined = path_combined

        self.path_counts   = self.path_analysis_root + "my_counts_data.csv"
        self.path_coldat   = self.path_analysis_root + "my_sample_info.csv"
        self.path_script   = self.path_analysis_root + "deseq2_analysis.R"
        self.path_trmnt1   = self.path_analysis_root + "deseq2_results_anticrispr_tr1.csv"
        self.path_trmnt2   = self.path_analysis_root + "deseq2_results_anticrispr_tr2.csv"
        self.path_ncount   = self.path_analysis_root + "deseq2_normalized_counts.csv"
        self.path_mrgd_1   = self.path_analysis_root + "deseq2_results_merged_tr1.csv"
        self.path_mrgd_2   = self.path_analysis_root + "deseq2_results_merged_tr2.csv"

    def generate_counts_and_col_data(self):

        if os.path.isfile(self.path_counts) and os.path.isfile(self.path_coldat):

            print("Counts and ColData already exist!")
            return

        else:

            df_combined = pd.read_csv(self.path_combined)
            dfc = df_combined.copy()

            p_columns = [f"p{n}" for n in self.plist]
            m_columns = [f"m{n}" for n in self.mlist]
            c_columns = [f"c{n}" for n in self.clist]

            cols = ['Geneid'] + p_columns + m_columns + c_columns

            df_counts = dfc[cols]
            df_counts.set_index("Geneid", inplace=True)
            df_counts.to_csv(self.path_counts, index_label=False)

            print(df_combined.head())
            print('\n')
            print(df_counts.head())

            # Now... make a dataframe whose index values are "cols". For those indexes that start with p (e.g., p0, p1), its ACR column value should be "tr1"
            # For those indexes that start with m (e.g., m0, m1), its ACR column value should be "tr2"
            # For those indexes that start with c (e.g., c0, c1), its ACR column value should be "control"

            df_coldata = pd.DataFrame(columns=['Sample', 'ACR'])
            df_coldata['Sample'] = cols[1:]
            df_coldata['ACR'] = ['tr1' if s.startswith('p') else 'tr2' if s.startswith('m') else 'untr' for s in cols[1:]]
            df_coldata.set_index('Sample', inplace=True)
            df_coldata.to_csv(self.path_coldat, index_label=False)
            print(df_coldata)

    def counts_sum_histogram(self):

        df = pd.read_csv(self.path_counts)

        df['col_sum'] = df.sum(axis=1)

        percentiles = [df['col_sum'].quantile(x / 100) for x in [10, 20, 30, 90]]

        percentile_dictionary = dict(zip(['10th', '20th', '30th', '90th'], percentiles))

        # plt.hist(df['col_sum'], bins=100, range=(df['col_sum'].min(), percentiles[3]))
        # for percentile in percentiles[:-1]: plt.axvline(x=percentile, color='black', linestyle='dashed')
        # plt.xlabel('col_sum')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of col_sum (up to 90th percentile)')
        # plt.suptitle(f"10th%: {percentiles[0]:.2f}, 20th%: {percentiles[1]:.2f}, 30th%: {percentiles[2]:.2f}, 90th%: {percentiles[3]:.2f}")
        # plt.savefig(self.path_analysis_root + f"counts_sum_histogram_{self.analysis_name}.png", dpi=300)  # plt.show()

        return percentile_dictionary

    def run(self):

        c0 = os.path.isfile(self.path_counts) and os.path.isfile(self.path_coldat)
        c1 = os.path.isfile(self.path_trmnt1) and os.path.isfile(self.path_trmnt2)

        if c0 and (not c1):

            pdict = self.counts_sum_histogram()
            epsilon = pdict[self.epsilon]
            print(self.epsilon, ": ",epsilon)

            # 1. Create the R script

            r_script = f"""
            suppressPackageStartupMessages(library("DESeq2"))
            suppressPackageStartupMessages(library(tidyverse))
            library(ggplot2)
            library(airway)

            countData <- read.csv('{self.path_counts}')
            colData <- read.csv('{self.path_coldat}')
            dds <- DESeqDataSetFromMatrix(countData = countData, colData = colData, design = ~ ACR)

            # pre-filtering: removing rows with low gene counts
            # keeping rows that have at least 10 reads total
            keep <- rowSums(counts(dds)) >= {epsilon}
            dds <- dds[keep,]

            dds$ACR <- relevel(dds$ACR, ref = "untr")
            dds <- DESeq(dds)
            res <- results(dds)

            res_tr1 <- results(dds, contrast = c("ACR", "tr1", "untr"))
            write.csv(res_tr1, '{self.path_trmnt1}')

            res_tr2 <- results(dds, contrast = c("ACR", "tr2", "untr"))
            write.csv(res_tr2, '{self.path_trmnt2}')

            # This saves the normalized counts that were used in the analysis
            norm_counts <- counts(dds, normalized=TRUE)
            write.csv(norm_counts,'{self.path_ncount}')
            """

            # 2. Save the script

            r_script = textwrap.dedent(r_script)

            with open(self.path_script, 'w') as f: f.write(r_script)

            # 3. Run the script.

            subprocess.check_call(f"Rscript {self.path_script}", shell=True)

        else:

            print("Analysis already done!")

    def generate_merged_results(self):

        if not os.path.isfile(self.path_mrgd_1):

            df1 = pd.read_csv(self.path_combined)
            df2 = pd.read_csv(self.path_trmnt1)
            df2 = df2.rename(columns={'Unnamed: 0': 'Geneid'})
            df2 = df2.merge(df1[['Geneid', 'Products']], on='Geneid', how='left')
            df2.to_csv(self.path_mrgd_1)

        else:

            print("Merged 1 results already exist!")

        if not os.path.isfile(self.path_mrgd_2):

            df1 = pd.read_csv(self.path_combined)
            df2 = pd.read_csv(self.path_trmnt2)
            df2 = df2.rename(columns={'Unnamed: 0': 'Geneid'})
            df2 = df2.merge(df1[['Geneid', 'Products']], on='Geneid', how='left')
            df2.to_csv(self.path_mrgd_2)

        else:

            print("Merged 2 results already exist!")

    def volcano_plot(self):

        def plot(ax, df_trna, df_tmsr, df_othr, title):

            p = 'padj' # 'pvalue' or 'padj'

            ax.scatter(x=df_othr['log2FoldChange'], y=-np.log10(df_othr[p]), c='blue', alpha=0.5, s=18)
            ax.scatter(x=df_trna['log2FoldChange'], y=-np.log10(df_trna[p]), c='red', alpha=0.8, s=18)
            ax.scatter(x=df_tmsr['log2FoldChange'], y=-np.log10(df_tmsr[p]), c='black', alpha=0.8, s=24)

            ax.set_xlabel("$log_{2}$ fold change", size = 15)
            ax.set_ylabel("- $log_{10}$ (padj)", size = 15)
            ax.set_title(title)
            ax.title.set_size(12)

            ax.set_xlim(-8, 8)
            ax.set_ylim(0, 10)

            # ax.axvline(x=0, c='black', alpha=0.5, zorder=0, ls='--')
            ax.axvline(x=-1, c='black', alpha=0.5, zorder=0, ls='--')
            ax.axvline(x=1, c='black', alpha=0.5, zorder=0, ls='--')

            ax.axhline(y=-np.log10(0.05), c='black', alpha=0.5, zorder=0, ls='--')

        pdict = self.counts_sum_histogram()
        epsilon = pdict[self.epsilon]
        print(self.epsilon, ": ",epsilon)

        df1 = pd.read_csv(self.path_mrgd_1)
        df2 = pd.read_csv(self.path_mrgd_2)

        df1_trna, df1_tmsr, df1_othr = df_trna_tmsr_othr_seperator(df1)
        df2_trna, df2_tmsr, df2_othr = df_trna_tmsr_othr_seperator(df2)

        fig, ax = plt.subplots(nrows= 2, ncols=1, figsize=(5, 10))

        plot(ax[0], df1_trna, df1_tmsr, df1_othr, f'MP16 vs WT - {self.analysis_name}')
        plot(ax[1], df2_trna, df2_tmsr, df2_othr, f'me6 vs WT - {self.analysis_name}')

        pl = f"ACR data: {repr(self.plist)}"  # .ljust(35)
        ml = f"No-ACR data: {repr(self.mlist)}"  # .ljust(35)
        cl = f"WT data: {repr(self.clist)}"  # .ljust(35)
        ep = f"epsilon: {self.epsilon}->{round(epsilon, 3)}"  # .ljust(35)

        txt= self.get_trna_stats(df1_trna)

        fig.suptitle(pl + " | " + ml + "\n" + cl + " | " + ep + "\n" + txt + "\n", fontsize=8)

        plt.savefig(self.path_analysis_root + f"volcano_plot_{self.analysis_name}.png", dpi=300) # plt.show()

    def MA_plot(self):

        def plot(ax, df_trna, df_othr, title):

            ax.scatter(x=df_othr['baseMean'], y=df_othr['log2FoldChange'], c='blue', alpha=0.5, s=18)
            ax.scatter(x=df_trna['baseMean'], y=df_trna['log2FoldChange'], c='red', alpha=0.8, s=18)

            ax.set_xlabel("baseMean", size = 15)
            ax.set_ylabel("$log_{2}$ fold change", size = 15)
            ax.set_title(title)
            ax.title.set_size(12)

            ax.set_xlim(0, 5000)
            ax.set_ylim(-3, 3)

            ax.axhline(y=0, c='black', alpha=0.5, zorder=0, ls='--')

        df1 = pd.read_csv(self.path_mrgd_1)
        df2 = pd.read_csv(self.path_mrgd_2)

        mask1 = (df1['Products'].str.startswith('tRNA-')) & (df1['Products'].str.len() < 10)
        mask2 = (df2['Products'].str.startswith('tRNA-')) & (df2['Products'].str.len() < 10)

        df1_trna = df1[mask1]
        df1_othr = df1[~mask1]
        df2_trna = df2[mask2]
        df2_othr = df2[~mask2]

        # 1. Plotting the volcano plot

        fig, ax = plt.subplots(nrows= 2, ncols=1, figsize=(5, 10))

        plot(ax[0], df1_trna, df1_othr, f'MP16 vs WT - {self.analysis_name}')
        plot(ax[1], df2_trna, df2_othr, f'me6 vs WT - {self.analysis_name}')
        plt.show()

    # -------

    @staticmethod
    def get_trna_stats(df_trna):

        mask = df_trna['padj'] <= 0.05
        df_filtered = df_trna[mask]

        n_sig1  = len(df_filtered)
        mean1   = df_filtered['log2FoldChange'].mean()
        median1 = df_filtered['log2FoldChange'].median()
        std1    = df_filtered['log2FoldChange'].std()

        n_sig2  = len(df_trna)
        mean2   = df_trna['log2FoldChange'].mean()
        median2 = df_trna['log2FoldChange'].median()
        std2    = df_trna['log2FoldChange'].std()

        st1 = f"Quadrant -> mean: {mean1:.2f} | median: {median1:.2f} | std: {std1:.2f} | n_sig: {n_sig1}"
        st2 = f"All      -> mean: {mean2:.2f} | median: {median2:.2f} | std: {std2:.2f} | n_sig: {n_sig2}"

        return st1 + '\n' + st2

    def report(self):

        df = pd.read_csv(self.path_mrgd_1)
        mask = (df['Products'].str.startswith('tRNA-')) & (df['Products'].str.len() < 10)
        df_trna = df[mask]
        df_othr = df[~mask]
        print(f'ANALYSIS: DESEQ2 -> merged_tr1.csv -> length = {len(df)}')
        print(df_trna.sort_values(by='baseMean', ascending=False))

    def analyse_normalized_counts(self):

        df1 = pd.read_csv(self.path_combined)
        df2 = pd.read_csv(self.path_ncount)
        df2 = df2.rename(columns={'Unnamed: 0': 'Geneid'})
        df2 = df2.merge(df1[['Geneid', 'Products']], on='Geneid', how='left')

        # Extracting columns starting with "p"
        p_columns = [col for col in df2.columns if col.startswith('p')]
        dfp = df2[['Geneid', 'Products'] + p_columns]

        # mask tRNAs
        mask = (dfp['Products'].str.startswith('tRNA-')) & (dfp['Products'].str.len() < 10)
        dfp = dfp[mask]

        # add a colum to dfp_trna that is true if the 'p5' column is the smallest among all p columns.
        dfp['p5_is_min'] = dfp.apply(lambda row: row['p5'] == min(row[p_columns]), axis=1)

        # count and print the boolean "True" values in the 'p5_is_min' column.
        print(dfp['p5_is_min'].value_counts())
        print(dfp)
        # df2.to_csv(self.path_mrgd_1)

class ProcessorAN2:

    path_gb_root = "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/genbank_files/"

    path_gb = {"LM1680_gff": path_gb_root + "LM1680.gff",
               "LM1680": path_gb_root + "LM1680.gb",
               "Twillingate": path_gb_root + "Twillingate.gb",
               "phiIBB": path_gb_root + "phiIBB.gb"}

    @staticmethod
    def command1(self, data_dic):

        path_cas = "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/pCRISPR-Cas.fasta"

        for dset in data_dic:

            for path in dset[1:]:
                # copy_file(path_cas, path_root + path + "queries/pCRISPR-Cas.fasta")
                # os.remove(path_root + path + "dvars.json")
                pass

    # This one parses log files and reads bowtie2 alignment rate and number of mapped reads. Makes a table.
    @staticmethod
    def command3(self, data_mat):

        rows_rate = list()
        path_gb_root = "/media/yourusername/Samsung980PRO/biodata2/anticrispr2/genbank_files/"

        for i in [0, 1, 2, 3, 4]:

            paths = [data_mat["acr_plus_phage"][i], data_mat["acr_minus_phage"][i], data_mat["negative_control"][i]]
            paths2 = [path_root + path for path in paths]

            for path in [path_root + data_mat["negative_control"][i]]:  # paths2:

                dvars = dvarGen(path)

                for name in ['LM1680', 'LM1680_pCRISPR']:  # dvars:

                    IGVM = MapperAN2(dvars[name], path_gb_root)

                    IGVM.printing_enabled = True
                    print(IGVM)
                    IGVM.execute_all()
                    IGVM.calculate_coverage()

                    rate, total = IGVM.read_overall_alignment_rate()
                    rows_rate.append([name, path.split('/')[-2], i, rate, total])

        ddf = pd.DataFrame(rows_rate, columns=['name', 'Path', 'i', 'Alignment Rate', 'Total Fragments'])
        print(ddf)

    # I forgot what this does
    @staticmethod
    def command5(self, data_mat, path_gb):

        def dvar_iterator(path_root, data_paths, path_gb, nm=''):

            for data_path in data_paths:

                path = path_root + data_path
                Utilities.print_colored(path, 'green')
                Utilities.print_colored(len(path) * '#' + '\n', 'green')
                time.sleep(0.2)
                dvars = dvarGen(path)

                if nm != '':

                    IGVM = MapperAN1(dvars[nm])
                    IGVM.printing_enabled = True
                    print(IGVM)
                    rr = IGVM.read_overall_alignment_rate()
                    print(f'\t\t[Overall Alignment Rate: {rr}%]\n\n')

                else:

                    for name in dvars:
                        IGVM = MapperAN1(dvars[name])
                        IGVM.printing_enabled = True
                        print(IGVM)
                        # IGVM.execute_all()
                        # IGVM.calculate_coverage()

                        IGVM.plot_whole_with_scipy_peaks(mode='Raw', show_inset=False, prominence=150,
                                                         path_gb=path_gb[name])
                        # name = 'RP62A'
                        # IGVM.gene_cov_analysis(path_gb[name])
                        # IGVM.pysam_exp()

        for n in [0, 1, 2, 3]: dvar_iterator(path_root, data_mat.iloc[n].tolist(), path_gb, nm='')

class tRNA:

    def __init__(self, query):
        self.id = query.id
        self.seq = query.seq

        res = self.parse_id()

        self.ltg = res[0]
        self.typ = res[1]
        self.iso = res[2]

    def __str__(self):

        return f"ltg:{self.ltg}, type: {self.typ}, iso: {self.iso}, seq: {self.seq}"

    def __eq__(self, other):
        return self.seq == other.seq

    def __hash__(self):
        return hash(str(self.seq))

    def parse_id(self):
        comps = self.id.split('_')

        ltg = f"{comps[2]}_{comps[3]}"
        typ = comps[4]
        iso = comps[5]

        return ltg, typ, iso

class tRNA_Processor:

    def __init__(self, path_queries, path_fasta_out, path_cm_file, path_ss_file, path_mappings):

        self.path_queries   = path_queries
        self.path_fasta_out = path_fasta_out
        self.path_infernal  = path_fasta_out.replace('.fasta', '.infernal')
        self.path_cm_file   = path_cm_file
        self.path_ss_file   = path_ss_file
        self.plot_name      = path_mappings.split('/')[-2]

        self.dvars = dvarGen(path_mappings, path_queries=path_queries)

    def step1_generate_unique_fasta_for_infernal(self):

        def print_trna_list(trna_list):

            print('------------------------------------\n\n')
            print(len(trna_list))
            for tr in trna_list:
                print(f"ltg:{tr.ltg}, type: {tr.typ}, iso: {tr.iso}, seq: {tr.seq}")

        trna_list = list()

        files = os.listdir(self.path_queries)

        files = [file for file in files if file.startswith('tRNA_') and file.endswith('.fasta')]

        for file in files:

            query = SeqIO.read(self.path_queries + file, "fasta")
            trna_list.append(tRNA(query))

        trna_list = list(set(trna_list))
        trna_list.sort(key=operator.attrgetter('typ', 'iso'), reverse=False)
        print_trna_list(trna_list)

        SecRecs = list()
        for trna in trna_list: SecRecs.append(SeqRecord(trna.seq, id=trna.id, description=''))

        SeqIO.write(SecRecs, self.path_fasta_out, "fasta")

    def step2_run_infernal(self):

        cmf = self.path_cm_file
        inp = self.path_fasta_out
        out = self.path_infernal

        cmd = f"cmalign -g --notrunc {cmf} {inp} > {out}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    def step3_map_coverage_and_plot(self):

        def match_coverage(t, almt, derivative=False):

            map_cov = list()
            map_ins = list()
            map_del = list()
            new_t   = list()

            i = 0
            for j, b in enumerate(almt):

                if b == '.':

                    map_cov.append(np.nan)
                    map_ins.append(1)
                    map_del.append(np.nan)

                elif b == '-':

                    map_cov.append(np.nan)
                    map_ins.append(np.nan)
                    map_del.append(1)

                else:

                    map_cov.append(i)
                    map_ins.append(np.nan)
                    map_del.append(np.nan)
                    i += 1

            for i in map_cov:

                if i is np.nan: new_t.append(i)
                else: new_t.append(t[i])

            new_t = np.array(new_t)/max(new_t)

            if derivative:

                new_t = np.abs(np.diff(new_t))
                new_t = np.insert(new_t, 0, 0)

            return new_t, map_cov, map_ins, map_del

        def loop_overlay(ax, x1, x2):

            color = "black"

            # ax.axvspan(x1, x2, zorder=2, alpha=0.1, edgecolor='none', facecolor="red")
            ax.axvline(x=x1, color=color, alpha=1, linewidth=1, zorder=2)
            ax.axvline(x=x2, color=color, alpha=1, linewidth=1, zorder=2)

        def match_coverage_test():

            almt = 'CGTG---ATT..AA'
            covr = '999111111'

            new_t, map_cov, map_ins, map_del = match_coverage(covr, almt)
            print(len(new_t), new_t)
            print(len(map_cov), map_cov)
            print(len(map_ins), map_ins)
            print(len(map_del), map_del)

        def determine_insertion_indices():

            s = "GGGGCTA..TAGCTCAGCT-GGG--A......GAGCGCTTGCATGGCATGCAAGAG-------------------GTCAGCGGTTCGATC............CC.GCT.TA..GCTCCACCA"

            ind = list()
            for i, letter in enumerate(s):

                if letter == '.':
                    ind.append(i)

            print(ind)

        def get_seq_dict_v0():

            cons_line = None
            rf_line = None
            SeqDict = OrderedDict()

            # Read Fasta File & populate SeqDict  ------------------------

            for record in SeqIO.parse(self.path_fasta_out, "fasta"):
                SeqDict[record.id] = [record, None]

            # Parse Infernal File & write to SeqDict  ------------------------

            lines = read_file(self.path_infernal, mode='lines')

            for i, line in enumerate(lines):

                if line.startswith('tRNA_'):

                    name, almt = line.split()

                    SeqDict[name][1] = almt.upper().replace('U',
                                                            'T')  # .replace('.', '').replace('-', '') # To standardize

                elif line.startswith('#=GC SS_cons'):

                    cons_line = line.split()[2]

                elif line.startswith('#=GC RF'):

                    rf_line = line.split()[2]

                else:

                    pass

            # for k,v in SeqDict.items(): print(k, '\t', v[0].seq == v[1]) # TEST IF the two datasets match.

            return SeqDict, cons_line, rf_line

        derivative_mode = True
        shading_mode = 'auto'
        p = {'fw': 8,  'fh': 6,  'dpi': 200, 'fsize': 7, 'w_spine': 1.0, 'tick_width': 1.0, 'tick_size': 2, 'tick_labelsize': 7 , 'tick_pad': 3.0}

        Data = list()

        # Remove extremely low coverage tRNAs from consideration
        remove_list = ['tRNA_Gly_K6U01_RS13050_Gly_GCC_neg',
                       'tRNA_Ile_K6U01_RS21565_Ile_CAT_neg',
                       'tRNA_Val_K6U01_RS00560_Val_GAC_pos',
                       'tRNA_Val_K6U01_RS06935_Val_TAC_pos',
                       'tRNA_Tyr_K6U01_RS21740_Tyr_GTA_pos',
                       'tRNA_Leu_K6U01_RS09930_Leu_CAG_neg',
                       'tRNA_Thr_K6U01_RS21750_Thr_GGT_pos',
                       'tRNA_Met_K6U01_RS16025_Met_CAT_pos']

        PIN = ParseInfernal(self.path_infernal, remove_list)
        PSS = ParseSS(self.path_ss_file)

        print(PIN)

        # PREPARE DATA ---------------------------------------------------

        for name in PIN:


            IGVM = MapperAN1(self.dvars[name])
            IGVM.attach_parsed_ss(PSS)

            x, _, t, _ = IGVM.extract_xyt_and_rpm()
            t = np.nan_to_num(t)

            max_cov = max(t)

            if max_cov >= 2000:

                comps = name.split('_')
                # label = f"{comps[4]}:{comps[5]}:{max_cov}"  # type and isotype
                label = f"{comps[4]}:{comps[5]}:{max_cov}-{comps[2]}_{comps[3]}"  # type and isotype

                # label = f"{comps[4]}:{comps[5]}"  # type and isotype

                new_t, map_cov, map_ins, map_del = match_coverage(t, PIN[name], derivative=derivative_mode)
                Data.append([label, new_t, map_ins, map_del])

        Data.sort(key = lambda x: x[0], reverse=True)

        y_labels =       [d[0] for d in Data]
        cov = np.asarray([d[1] for d in Data])
        dlt = np.asarray([d[3] for d in Data])

        del_indices = [7, 8, 26, 27, 28, 29, 30, 31, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 104, 108, 111, 112]
        cov = np.delete(cov, del_indices, axis=1)
        dlt = np.delete(dlt, del_indices, axis=1)

        N = len(cov[0])
        x = np.linspace(0, N - 1, N)
        y = np.linspace(0, len(Data) - 1, len(Data))

        X, Y = np.meshgrid(x, y)

        # PLOT ----------------------------------------------------------

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(p['fw'], p['fh']), dpi=p['dpi'])
        ax.set_zorder(1)

        pc = ax.pcolormesh(X, Y, cov, edgecolor='none', cmap='YlOrRd', shading=shading_mode, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1))
        # pc = ax.pcolormesh(X, Y, cov, edgecolor='none', cmap='Greys', shading=shading_mode, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1))
        # pc = ax.pcolormesh(X, Y, cov, edgecolor='none', cmap='Greys', shading=shading_mode)

        cbar = fig.colorbar(pc, pad=0.03, drawedges=False, format='%.1f')
        cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        ax.scatter(X, Y, s=1* dlt, marker = '+', color='black')

        loop_overlay(ax, x1=13-1, x2=23+1)
        loop_overlay(ax, x1=34-1, x2=40+1)
        loop_overlay(ax, x1=55-1, x2=59+1)
        loop_overlay(ax, x1=75-1, x2=81+1)

        cbar.ax.set_ylabel('|Derivative of Normalized Raw Coverage|', rotation=90, fontsize=p['fsize'])

        cbar.ax.tick_params(width=p['tick_width'], size=p['tick_size'], labelsize=p['tick_labelsize'], pad=p['tick_pad'])
        ax.tick_params(width=p['tick_width'], size=p['tick_size'], labelsize=p['tick_labelsize'], pad=p['tick_pad']) # width controls line width, size control the length of the tick marks.

        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, font="DejaVu Sans Mono", fontsize=p['fsize'])

        CL = np.array(list(PIN.cons_line))
        RF = np.array(list(PIN.rf_line))
        CL = np.delete(CL, del_indices)
        RF = np.delete(RF, del_indices)

        x_labels = [f"{c}\n{r}" for c, r in zip(CL, RF)]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, font="DejaVu Sans Mono", fontsize=8)

        from matplotlib.ticker import AutoMinorLocator

        # ax.yaxis.minorticks_on()
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # ax.grid(b=True, which='minor', axis='y', color='#000000', linestyle='-', alpha=0.3) # Before python 3.11
        ax.grid(which='minor', axis='y', color='#000000', linestyle='-', alpha=0.3)

        print(self.plot_name)
        fig.suptitle(self.plot_name, fontsize=p['fsize'])
        plt.tight_layout()

        plt.show()

class SunSetAnalysis:

    def __init__(self, pvar):

        self.path_mpg = pvar['mapping']
        self.path_gnb = pvar['genbank']
        self.path_all = pvar['trnaall']
        self.path_unq = pvar['uniqfst']
        self.path_qry = pvar['queries']
        self.path_pss = pvar['pssfile']
        self.path_inf = pvar['infernl']
        self.path_cmf = pvar['cmffile']

        self.pvar  = pvar
        self.dvars = dvarGen(self.path_mpg, path_queries=self.path_qry) # PSS = ParseSS(path_pss)

    def step0_create_tRNA_mappings(self):

        PSS = ParseSS(self.path_pss)

        for name in self.dvars:
            IGVM = MapperAN1(self.dvars[name])
            IGVM.printing_enabled = True  # print(IGVM)
            IGVM.execute_all()
            IGVM.calculate_coverage()

            IGVM.attach_parsed_ss(PSS)
            IGVM.analyze_start_ends_without_peak_detection(self.path_mpg + "/figures/")

    def step1_export_tRNAs_from_genbank(self, mode='all', padding=0):

        record: SeqRecord = SeqIO.read(self.path_gnb, "gb")  # self.record is of the type: <class 'Bio.SeqRecord.SeqRecord'>
        SecRecs = list()
        required_keys = {'locus_tag', 'product', 'anticodon'}

        for feature in record.features:

            quals = feature.qualifiers
            qual_set = set(quals.keys())

            # if 'product' in qual_set: print(quals['product'][0], qual_set)

            if required_keys.issubset(qual_set) and 'pseudo' not in quals.keys():

                ltag = quals['locus_tag'][0]
                prod = quals['product'][0]
                anticodon = quals['anticodon'][0]

                print()
                print(f'\tfound {prod} {anticodon} - {qual_set}')

                m = re.search(r'aa:([a-zA-Z]{3,5}),\s*seq:([a-z]{3})', anticodon)

                isotype = m.group(1)
                anticodon_seq = m.group(2)

                id_string = f"{prod}_{ltag}_{isotype}_{anticodon_seq.upper()}"
                id_string = id_string.replace('-', '_').replace(' ', '_')

                seq: Seq = record.seq[feature.location.start - padding: feature.location.end + padding]

                if feature.location.strand == -1:
                    seq = seq.reverse_complement()
                    strand = "_neg"
                else:
                    strand = "_pos"

                SecRecs.append(SeqRecord(seq, id=id_string + strand, description=''))

        # Write to file

        if mode == 'all':

            SeqIO.write(SecRecs, self.path_all, "fasta")

        else:

            dn = os.path.dirname(self.path_all)
            for SecRec in SecRecs: SeqIO.write(SecRec, dn + '/' + SecRec.id + '.fasta', "fasta")

    def step1_save_individual_trnas(self):

        records: SeqRecord = SeqIO.parse(self.path_all, "fasta")

        for SecRec in records:
            print(SecRec.id)
            SeqIO.write(SecRec, self.path_qry + SecRec.id + '.fasta', "fasta")

    def step2_run_tRNAscan_SE(self):

        pass

    def step3_create_unique_tRNA_fasta_for_infernal(self):

        def print_trna_list(trna_list):

            print('------------------------------------\n\n')
            print(len(trna_list))
            for tr in trna_list: print(tr)

        trna_list = list()

        files = os.listdir(self.path_qry)

        files = [file for file in files if file.startswith('tRNA_') and file.endswith('.fasta')]

        for file in files:
            query = SeqIO.read(self.path_qry + file, "fasta")
            trna_list.append(tRNA(query))

        trna_list = list(set(trna_list))
        trna_list.sort(key=operator.attrgetter('typ', 'iso'), reverse=False)

        print(len(trna_list), len(files))
        print_trna_list(trna_list)

        SecRecs = list()
        for trna in trna_list: SecRecs.append(SeqRecord(trna.seq, id=trna.id, description=''))

        SeqIO.write(SecRecs, self.path_unq, "fasta")

    def step4_run_infernal(self):

        cmd = f"cmalign -g --notrunc {self.path_cmf} {self.path_unq} > {self.path_inf}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    def step5_plot_sunset_diagram(self):

        plot_name     = self.pvar['plot_nm']
        cov_threshold = self.pvar['min_cov']
        del_indices   = self.pvar['del_ind'] 
        remove_list   = self.pvar['removel'] 

        def match_coverage(t, almt, derivative=False):

            map_cov = list()
            map_ins = list()
            map_del = list()
            new_t = list()

            i = 0
            for j, b in enumerate(almt):

                if b == '.':

                    map_cov.append(np.nan)
                    map_ins.append(1)
                    map_del.append(np.nan)

                elif b == '-':

                    map_cov.append(np.nan)
                    map_ins.append(np.nan)
                    map_del.append(1)

                else:

                    map_cov.append(i)
                    map_ins.append(np.nan)
                    map_del.append(np.nan)
                    i += 1

            for i in map_cov:

                if i is np.nan:
                    new_t.append(i)
                else:
                    new_t.append(t[i])

            new_t = np.array(new_t) / max(new_t)

            if derivative:
                new_t = np.abs(np.diff(new_t))
                new_t = np.insert(new_t, 0, 0)

            return new_t, map_cov, map_ins, map_del

        def loop_overlay(ax, x1, x2):

            color = "black"

            # ax.axvspan(x1, x2, zorder=2, alpha=0.1, edgecolor='none', facecolor="red")
            ax.axvline(x=x1, color=color, alpha=1, linewidth=1, zorder=2)
            ax.axvline(x=x2, color=color, alpha=1, linewidth=1, zorder=2)

        def extract_loop_boundaries(annotation):

            loop_boundaries = []
            loop_regions = re.findall(r'<[_.]+>', annotation)

            for loop in loop_regions:
                start_index = annotation.index(loop)
                end_index = start_index + len(loop) - 1
                loop_boundaries.append((start_index, end_index))

                annotation = annotation.replace(loop, '.' * len(loop), 1)

            return loop_boundaries

        derivative_mode = True
        shading_mode = 'auto'
        p = {'fw': 8, 'fh': 6, 'dpi': 200, 'fsize': 7, 'w_spine': 1.0, 'tick_width': 1.0, 'tick_size': 2,
             'tick_labelsize': 7, 'tick_pad': 3.0}

        Data = list()

        PIN = ParseInfernal(self.path_inf, remove_list)
        PSS = ParseSS(self.path_pss)

        print(PIN)

        # PREPARE DATA ---------------------------------------------------

        for name in PIN:

            IGVM = MapperAN1(self.dvars[name])
            IGVM.attach_parsed_ss(PSS)

            x, _, t, _ = IGVM.extract_xyt_and_rpm()
            t = np.nan_to_num(t)

            max_cov = max(t)

            if max_cov >= cov_threshold:
                comps = name.split('_')
                label = f"{comps[4]}:{comps[5]}:{max_cov}-{comps[2]}_{comps[3]}"  # type and isotype

                new_t, map_cov, map_ins, map_del = match_coverage(t, PIN[name], derivative=derivative_mode)
                Data.append([label, new_t, map_ins, map_del])

        Data.sort(key=lambda x: x[0], reverse=True)

        y_labels = [d[0] for d in Data]
        cov = np.asarray([d[1] for d in Data])
        dlt = np.asarray([d[3] for d in Data])

        cov = np.delete(cov, del_indices, axis=1)
        dlt = np.delete(dlt, del_indices, axis=1)

        N = len(cov[0])
        x = np.linspace(0, N - 1, N)
        y = np.linspace(0, len(Data) - 1, len(Data))

        X, Y = np.meshgrid(x, y)

        # PLOT ----------------------------------------------------------

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(p['fw'], p['fh']), dpi=p['dpi'])
        ax.set_zorder(1)

        pc = ax.pcolormesh(X, Y, cov, edgecolor='none', cmap='YlOrRd', shading=shading_mode, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1))
        # pc = ax.pcolormesh(X, Y, cov, edgecolor='none', cmap='Greys', shading=shading_mode, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1))
        # pc = ax.pcolormesh(X, Y, cov, edgecolor='none', cmap='Greys', shading=shading_mode)

        cbar = fig.colorbar(pc, pad=0.03, drawedges=False, format='%.1f')
        cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        ax.scatter(X, Y, s=1 * dlt, marker='+', color='black')

        cbar.ax.set_ylabel('|Derivative of Normalized Raw Coverage|', rotation=90, fontsize=p['fsize'])

        cbar.ax.tick_params(width=p['tick_width'], size=p['tick_size'], labelsize=p['tick_labelsize'], pad=p['tick_pad'])
        ax.tick_params(width=p['tick_width'], size=p['tick_size'], labelsize=p['tick_labelsize'], pad=p['tick_pad'])  # width controls line width, size control the length of the tick marks.

        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, font="DejaVu Sans Mono", fontsize=p['fsize'])

        CL = np.array(list(PIN.cons_line))
        RF = np.array(list(PIN.rf_line))
        CL = np.delete(CL, del_indices)
        RF = np.delete(RF, del_indices)

        print(CL)
        pairs = extract_loop_boundaries(''.join(CL)) # PIN.cons_line

        for pair in pairs: loop_overlay(ax, x1=pair[0], x2=pair[1])

        x_labels = [f"{c}\n{r}" for c, r in zip(CL, RF)]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, font="DejaVu Sans Mono", fontsize=8)

        from matplotlib.ticker import AutoMinorLocator

        # ax.yaxis.minorticks_on()
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(which='minor', axis='y', color='#000000', linestyle='-', alpha=0.3)

        print(plot_name)
        fig.suptitle(plot_name, fontsize=p['fsize'])
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":

    path_gb_root    = path_root + "genbank_files/"
    path_deseq      = path_root + "deseq_analysis/"

    path_combined   = path_deseq + "df_combined.csv"
    path_normalized = path_deseq + "df_normalized.csv"

    pr = path_root
    data_dic = [[0, pr + "mappings_acr_plus_phage_rep0/", pr + "mappings_acr_mns_phage_rep0/", pr + "mappings_negative_contrl_rep0/"],
                [1, pr + "mappings_acr_plus_phage_rep1/", pr + "mappings_acr_mns_phage_rep1/", pr + "mappings_negative_contrl_rep1/"],
                [2, pr + "mappings_acr_plus_phage_rep2/", pr + "mappings_acr_mns_phage_rep2/", pr + "mappings_negative_contrl_rep2/"],
                [3, pr + "mappings_acr_plus_phage_rep3/", pr + "mappings_acr_mns_phage_rep3/", pr + "mappings_negative_contrl_rep3/"],
                [4, pr + "mappings_acr_plus_phage_rep4/", pr + "mappings_acr_mns_phage_rep4/", pr + "mappings_negative_contrl_rep4/"],
                [5, pr + "mappings_acr_plus_phage_rep5/", pr + "mappings_acr_mns_phage_rep5/", pr + "mappings_negative_contrl_rep5/"],
                [6, pr + "mappings_acr_plus_phage_rep6/", pr + "mappings_acr_mns_phage_rep6/", pr + "mappings_negative_contrl_rep6/"],
                [7, pr + "mappings_acr_plus_phage_rep7/", pr + "mappings_acr_mns_phage_rep6/", pr + "mappings_negative_contrl_rep7/"]]

    data_mat = pd.DataFrame(data_dic, columns=["i", "acr_plus_phage", "acr_minus_phage", "negative_control"])
    data_mat.set_index('i', inplace=True)  # print(data_mat)






