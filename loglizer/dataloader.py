"""
The interface to load log datasets. The datasets currently supported include
HDFS and Spirit

"""

import pandas as pd
import os
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict
from datetime import timedelta


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform', TB=False):
    if split_type == 'uniform' and y_data is not None:
        
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])

        if TB==True:
            x_train = np.concatenate((x_pos[0:train_pos], x_neg[0:train_neg]), axis=0) #np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
            y_train = np.concatenate((y_pos[0:train_pos], y_neg[0:train_neg]), axis=0) 
            x_test = np.concatenate((x_pos[train_pos:], x_neg[train_neg:]), axis=0) #np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
            y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        else:
            x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
            y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
            x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
            y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
            
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle 
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    '''
    indexes = shuffle(np.arange(x_test.shape[0]))
    x_test = x_test[indexes]
    if y_test is not None:
        y_test = y_test[indexes]
    '''    
    return (x_train, y_train), (x_test, y_test)

###############
###############
def prinf_contentData(dataset):
    for index, row in dataset.head().iterrows():
        print(row.to_string())  # Converts the row into a readable string format
        print("-" * 10)  # Separator for readability
##############
##############

def load_SPIRIT(log_file, event_traces, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0, TB=False, WV=False):
    if log_file.endswith('.csv'):
        print(f"Loading {log_file}")
        use_cols = ['Label', 'Timestamp', 'EventId', 'EventTemplate']  # Replace with your actual needed columns
        struct_log = pd.read_csv(log_file, engine='c',
                    na_filter=False, memory_map=True, usecols=use_cols)

        
        print(len(struct_log))
        
        struct_log['timestamp'] = pd.to_datetime(struct_log['Timestamp'] - 8*3600, unit='s', utc=True)              
        
        #count = struct_log['Label'].isin(['R_HDA_NR', 'R_HDA_STAT', 'N_PBS_CHK', 'R_GM_LANAI', 'N_PBS_PRE', 'N_PBS_BAIL', 'N_AUTH', 'N_PBS_BFD1', 'N_PBS_EPI' ,'R_EXT_CCISS', 'N_NFS', 'R_NMI', 'N_GM_MAP', 'N_PBS_CON1', 'R_GM_PAR2', 'R_GM_PAR4', 'R_GM_PAR1', 'N_OOM', 'R_GM_PAR3', 'R_NMI1']).sum()
        #print("Count:", count)

        count = (struct_log['Label'] != '-').sum()
        print("Count of values not equal to '-':", count)
        
        # Sort logs chronologically
        struct_log = struct_log.sort_values(by='timestamp').reset_index(drop=True)
        
        # Define sliding window parameters
        window_size = timedelta(minutes=60)   # 1-hour window
        #step_size = timedelta(minutes=30)  # 30-minute overlap for example (adjustable) 
        step_size = timedelta(minutes=60)   # To use the fixed window strategy to group log data 
        
        # Initialize variable
        start_time = struct_log['timestamp'].min()
        end_time = struct_log['timestamp'].max()
        
        session_data = []
        session_id = 0

        # Extract base filename without extension
        # base_name = os.path.splitext(os.path.basename(log_file))[0]  

        # doc = os.path.join(os.path.dirname(log_file), f'{base_name}.feather')

        # if not os.path.exists(doc):
        # Apply sliding window
        while start_time < end_time:
            window_end = start_time + window_size
            mask = (struct_log['timestamp'] >= start_time) & (struct_log['timestamp'] < window_end)
            window_logs = struct_log[mask].copy()
            
            if not window_logs.empty:
                window_logs['Session_ID'] = session_id
                session_data.append(window_logs)
                session_id += 1
            
            # Slide the window
            start_time += step_size
        
        # Combine all session windows
        grouped_df = pd.concat(session_data, ignore_index=True)

            #grouped_df.to_feather(doc)
            
        #grouped_df = pd.read_feather(doc)
        
        # Save or display the result
        #grouped_df.to_csv("spirit_sliding_window.csv", index=False)
        print(grouped_df[['timestamp', 'Session_ID']])

        if WV:
            eventType = 'EventTemplate'
        else:
            eventType = 'EventId'
        
        # Group EventTemplate by sessionID into a list
        #session_template_sequences = grouped_df.groupby('Session_ID')['EventTemplate'].apply(list).reset_index()
        session_template_sequences = grouped_df.groupby('Session_ID')[eventType].apply(list).reset_index()
        
        # Rename columns for clarity (optional)
        #session_template_sequences.columns = ['Session_ID', 'EventTemplateSequence']
        session_template_sequences.columns = ['Session_ID', eventType + 'Sequence']

        # Compute average sequence length
        average_sequence_length = session_template_sequences[eventType + 'Sequence'].apply(len).mean()
        
        print(f"Average sequence length: {average_sequence_length:.2f}")
        
        # Determine the label per session:
        # Assign 0 if all Label values for that Session_ID are '-', else 0
        # Assign 0 if all Label values for that Session_ID are '-', else 1
        session_labels = grouped_df.groupby('Session_ID')['Label'].apply(
            lambda x: int(not all(v == '-' for v in x))
        ).reset_index(name='Label')

        # Merge labels into the session_template_sequences
        session_template_sequences = session_template_sequences.merge(session_labels, on='Session_ID')

        print(session_template_sequences)        
        
        (x_train, y_train), (x_test, y_test) = _split_data(session_template_sequences[eventType + 'Sequence'].values, 
                session_template_sequences['Label'].values, train_ratio, split_type)

        #print(len(x_train))
        #print(len(x_test))
        #import sys
        #sys.exit(0)      
        
        
        #print(y_train)
        #print(y_test)
        #import sys
        #sys.exit()
        
        return (x_train, y_train), (x_test, y_test)
        
def load_HDFS(log_file, event_traces, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0, TB=False, WV=False):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file, allow_pickle=True)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)
        
        log_traces = pd.read_csv(event_traces, engine='c',
                na_filter=False, memory_map=True)
        
        data_df = log_traces[['BlockId','Label', 'Features']].rename(columns={'Features': 'EventSequence'}).copy()
        data_df['EventSequence'] = data_df['EventSequence'].apply(lambda x: x.strip("[]").split(","))
        
        ##### 
        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
            print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)
        #####
    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        print(f"Loading {log_file}")
        use_cols = ['Date', 'Time', 'EventId', 'Content', 'EventTemplate' , 'EventId']  # Replace with your actual needed columns
        struct_log = pd.read_csv(log_file, engine='c',
                    na_filter=False, memory_map=True, usecols=use_cols)
        print(struct_log)
        
        log_traces = pd.read_csv(event_traces, engine='c',
                na_filter=False, memory_map=True)
        ###############
        ###############
        #print("struct_log")
        #dataset = struct_log.copy()
        #prinf_contentData(dataset)
        ##############
        ##############
        
        ####
        
        # Extract base filename without extension
        base_name = os.path.splitext(os.path.basename(log_file))[0]  # 'HDFS_100k.log_structured'

        bnw2v=base_name+'Word2Vec'
        
        doc = os.path.join(os.path.dirname(log_file), f'{base_name}.feather')
        
        doc2 = os.path.join(os.path.dirname(log_file), f'{bnw2v}.feather')
        
        if not os.path.exists(doc):
            data_dict = OrderedDict()
            for idx, row in struct_log.iterrows():
                blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])

                blkId_set = set(blkId_list)

                for blk_Id in blkId_set:
                    if not blk_Id in data_dict:
                        data_dict[blk_Id] = []
                    data_dict[blk_Id].append(row['EventId']) # A sequence is all EventIds with the same blk_Id
                    #print(data_dict[blk_Id])

            data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
            # Save
            data_df.to_feather(doc)
            
        data_df = pd.read_feather(doc)
        
        if WV==True:
            if not os.path.exists(doc2):
                data_dict = OrderedDict()
                for idx, row in struct_log.iterrows():
                    blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])

                    blkId_set = set(blkId_list)

                    for blk_Id in blkId_set:
                        if not blk_Id in data_dict:
                            data_dict[blk_Id] = []
                        data_dict[blk_Id].append(row['EventTemplate']) # A sequence is all EventIds with the same blk_Id
                        #print(data_dict[blk_Id])

                data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
                # Save
                data_df.to_feather(doc2)
                
            data_df = pd.read_feather(doc2)    
        
        ####
        
        #print(data_dict['blk_3888635850409849568'],'------------------------------------------------')
        
        ###############
        ###############
        #prinf_contentData(data_df)
        #print("The number of sequences is:")
        #print(len(data_df['EventSequence']))
        ##############
        ##############
        
        if TB==True:
            # To do : consider date in computing session_time_diff and session_time_diff2 columns
            
            ####
            #struct_log['Time'] = struct_log['Time'].astype(str).str.zfill(6)
            struct_log['Time'] = struct_log['Time'].astype('string').str.zfill(6)
            struct_log['Date'] = struct_log['Date'].astype('string').str.zfill(6)  # Ensures YYMMDD

            ####
            
            # Combine Date and Time into a single datetime column
            struct_log['timestamp'] = pd.to_datetime(
                struct_log['Date'].astype(str) + ' ' + struct_log['Time'].astype(str),
                format='%y%m%d %H%M%S'
            )
            
            print(struct_log[['timestamp']])
            
            #struct_log['timestamp'] = pd.to_datetime(struct_log['Time'].astype(str), format='%H%M%S')
            
            struct_log['BlockId'] = struct_log['Content'].str.extract(r'(blk_-?\d+)', expand=False)
   
            struct_log['session_id'] = struct_log['BlockId'].astype('category').cat.codes  # Convert block ID to numeric
            
            #print(struct_log['session_id'])

            struct_log['session_time_diff'] = struct_log.groupby('BlockId')['timestamp'].diff().dt.total_seconds().fillna(0)
            
            struct_log['session_time_diff2'] = struct_log['timestamp'].diff().dt.total_seconds().fillna(0)
            
            block_time_diff_sum = struct_log.groupby('BlockId')['session_time_diff'].sum().reset_index()

            block_time_diff_sum2 = struct_log.groupby('BlockId')['session_time_diff2'].sum().reset_index()
            
            block_time_diff_sum.rename(columns={'session_time_diff': 'total_session_time_diff'}, inplace=True)
            
            block_time_diff_sum2.rename(columns={'session_time_diff2': 'total_session_time_diff2'}, inplace=True)

            #print(struct_log[['session_time_diff', 'BlockId']])
            
            #print(block_time_diff_sum)
                
            block_time_dict = block_time_diff_sum.set_index("BlockId")["total_session_time_diff"].to_dict()
            
            block_time_dict2 = block_time_diff_sum2.set_index("BlockId")["total_session_time_diff2"].to_dict()

            data_df["total_session_time_diff"] = data_df["BlockId"].map(block_time_dict)
            
            data_df["total_session_time_diff2"] = data_df["BlockId"].map(block_time_dict2)
            
            #######
            log_traces_dict = log_traces.set_index("BlockId")["Latency"].to_dict()
        
            data_df["Latency"] = data_df["BlockId"].map(log_traces_dict)
            #######
            
            print(data_df[['total_session_time_diff', 'total_session_time_diff2', 'Latency']])
            
            from sklearn.preprocessing import MinMaxScaler
            
            scaler = MinMaxScaler()
            
            data_df[['total_session_time_diff']] = scaler.fit_transform(data_df[['total_session_time_diff']])
            
            data_df[['total_session_time_diff2']] = scaler.fit_transform(data_df[['total_session_time_diff2']])
            
            data_df[['Latency']] = scaler.fit_transform(data_df[['Latency']])
            
            #print(data_df[['BlockId','total_session_time_diff', 'total_session_time_diff2']])

            
        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            
            ###############
            ###############
            #print("Labels:")
            #dataset = label_data.copy()
            #prinf_contentData(label_data)
            ##############
            ##############
            
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
            
            #print("data_df['EventSequence'].values:",data_df['EventSequence'].values)
            # Split train and test data
            if TB==True:
                (x_train, y_train), (x_test, y_test) = _split_data(data_df[['EventSequence' 
                ,'total_session_time_diff', 'total_session_time_diff2'  
                #, 'Latency'
                ]].values, 
                data_df['Label'].values, train_ratio, split_type,True)
            else:
                (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, train_ratio, split_type)

            '''
            ##########
            ##########
            print(data_df['EventSequence'][0])
            print(data_df['EventSequence'][1])
            print(data_df['EventSequence'][2])
            print(data_df['EventSequence'][3])
            print(data_df['EventSequence'][4])
            print(data_df['Label'][0])
            print(data_df['Label'][1])
            print(data_df['Label'][2])
            print(data_df['Label'][3])
            print(data_df['Label'][4])
                        
            print(y_test.shape[0])
            ##########
            ##########
            '''
            print(y_train.sum(), y_test.sum())
        
        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)
        
        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
            print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)
        
        #print(data_df.head())
        
        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test), data_df

def slice_hdfs(x, y, window_size):
    results_data = []
    print("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            '''
            # Ensure the array can hold strings
            slice = slice.astype(object)

            # Append padding
            slice = np.concatenate((slice, ["#Pad"] * (window_size - len(slice))))
            '''
            results_data.append([idx, slice, "#Pad", y[idx]])
    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    print("Slicing done, {} windows generated".format(results_df.shape[0]))
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]



def load_BGL(log_file, label_file=None, window='sliding', time_interval=60, stepping_size=60, 
             train_ratio=0.8):
    """  TODO

    """


def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = para['save_path']+'sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n'%inst_number)
        np.savetxt(sliding_file_path,start_end_index_list,delimiter=',',fmt='%d')
    else:
        print('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = [row[0] for row in event_mapping_data]
    event_num = len(list(set(event_mapping_data)))
    print('There are %d log events'%event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    event_count_matrix = np.zeros((inst_number,event_num))
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            event_index = event_mapping_data[k]
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels
