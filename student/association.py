# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
#        print('func_in associate')
        
        # the following only works for at most one track and one measurement
#        self.association_matrix = np.matrix([]) # reset matrix
#        self.unassigned_tracks = [] # reset lists
#        self.unassigned_meas = []
        len_track = len(track_list)
        len_meas = len(meas_list)
        print('len_track', len_track)
        print('len_meas', len_meas)
        
#        if len(meas_list) > 0:
#            self.unassigned_meas = [0]
#        if len(track_list) > 0:
#            self.unassigned_tracks = [0]
#        if len(meas_list) > 0 and len(track_list) > 0: 
#            self.association_matrix = np.matrix([[0]])
        
        self.unassigned_tracks = list(range(len_track))
        self.unassigned_meas = list(range(len_meas))
        self.association_matrix = np.asmatrix(np.inf * np.ones((len_track, len_meas)))

        for i in range(len_track): 
            track = track_list[i]
            for j in range(len_meas):
                meas = meas_list[j]
                dist = self.MHD(track, meas, KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i, j] = dist
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement
        # 以下は、多くても1つのトラックと1つの測定でのみ機能します
        update_track = 0
        update_meas = 0
        # find closest track and measurement for next update
        # 次の更新のために最も近いトラックと測定値を見つける
        ##### 関連付けマトリクスの要素の最小値が無限の場合、関連付け終了
        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan
        
        # get indices of minimum entry
        # 最小エントリのインデックスを取得
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]
        
        # delete row and column for next update
        # 次の更新のために行と列を削除します
        A = np.delete(A, ind_track, 0) 
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A

        # update this track with this measurement
        # この測定値でこのトラックを更新します
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove from list
        # remove this track and measurement from list
        # このトラックと測定値をリストから削除します
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        # check if measurement lies inside gate
        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD < limit:
            is_inside = True
        else:
            is_inside = False
        return is_inside

        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        # calc Mahalanobis distance
#        H = np.matrix([[1, 0, 0, 0],
#                       [0, 1, 0, 0]]) 
        x = track.x
        P = track.P
    
        H = meas.sensor.get_H(x)
        gamma =  KF.gamma(track, meas)
        S = KF.S(track, meas, H)
        MHD = gamma.T * S.I * gamma # Mahalanobis distance formula
        return MHD
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)