import struct
import numpy as np

def parse_mjl_logs(read_filename, skipamount):
    with open(read_filename, mode='rb') as file:
        fileContent = file.read()
    headers = struct.unpack('iiiiiii', fileContent[:28])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    nuserdata = headers[5]
    name_len = headers[6]
    name = struct.unpack(str(name_len) + 's', fileContent[28:28+name_len])[0]
    rem_size = len(fileContent[28 + name_len:])
    num_floats = int(rem_size/4)
    dat = np.asarray(struct.unpack(str(num_floats) + 'f', fileContent[28+name_len:]))
    recsz = 1 + nq + nv + nu + 7*nmocap + nsensordata + nuserdata
    if rem_size % recsz != 0:
        print("ERROR")
    else:
        dat = np.reshape(dat, (int(len(dat)/recsz), recsz))
        dat = dat.T

    time = dat[0,:][::skipamount] - 0*dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skipamount, :]
    qvel = dat[nq+1:nq+nv+1,:].T[::skipamount, :]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skipamount,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skipamount, :]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skipamount, :]
    sensordata = dat[nq+nv+nu+7*nmocap+1:nq+nv+nu+7*nmocap+nsensordata+1,:].T[::skipamount,:]
    userdata = dat[nq+nv+nu+7*nmocap+nsensordata+1:,:].T[::skipamount,:]

    data = dict(nq=nq,
               nv=nv,
               nu=nu,
               nmocap=nmocap,
               nsensordata=nsensordata,
               name=name,
               time=time,
               qpos=qpos,
               qvel=qvel,
               ctrl=ctrl,
               mocap_pos=mocap_pos,
               mocap_quat=mocap_quat,
               sensordata=sensordata,
               userdata=userdata,
               logName = read_filename
               )
    return data
