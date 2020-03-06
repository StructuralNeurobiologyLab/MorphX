import os
from tqdm import tqdm
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds
from morphx.data import basics

if __name__ == '__main__':
    tech_density = 1500
    bio_density = 50
    sample_num = 28000
    data_path = os.path.expanduser('~/loc_Bachelorarbeit/gt/')
    save_path = os.path.expanduser(f'~/loc_Bachelorarbeit/gt/samples/d{bio_density}/')

    ch = ChunkHandler(data_path, sample_num, density_mode=True, bio_density=bio_density, tech_density=tech_density,
                      specific=True)

    # for obj in ch.obj_names:
    #     full = None
    #     samples = []
    #     for ix in tqdm(range(ch.get_obj_length(obj))):
    #         chunk, idcs, bfs = ch[(obj, ix)]
    #         samples.append([chunk, bfs])
    #         if full is None:
    #             full = chunk
    #         else:
    #             full = clouds.merge_clouds([full, chunk])
    #     full.save2pkl(f'{save_path}{obj}_d{bio_density}.pkl')
    #     basics.save2pkl(samples, save_path, name=f'{obj}_d{bio_density}_samples')
