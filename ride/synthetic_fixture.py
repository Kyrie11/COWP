"""Only for unit tests. Synthetic rows MUST NOT be used as evidence or a paper figure."""
import argparse,json
from pathlib import Path

def rows():
    out=[{'type':'manifest','simulator':'CARLA','archetype':'convoy',
          'carla_server_version':'SYNTHETIC_NOT_REAL'}]
    for family in ('tm','behavior'):
        for profile in ('early_slow','late_slow','mild_early','mild_late'):
            bad=profile in ('early_slow','mild_early')
            init={role:{'xyz':[x,0,0],'speed_mps':9.}
                  for role,x in (('ego',20),('follower_a',0),('follower_b',-18))}
            out.append({'type':'trial','archetype':'convoy','anchor':0,'seed':7,
                        'family':family,'profile':profile,'initial_states':init,
                        'metrics':{'valid':True,'ego_progress_m':50.,
                                   'ego_collision':False,'non_ego_pair_collision':bad,
                                   'hard_brake':bad,'min_rear_pair_ttc_s':0.7 if bad else 5.0}})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    d=Path(a.output);d.parent.mkdir(parents=True,exist_ok=True)
    d.write_text(''.join(json.dumps(q)+'\n' for q in rows()))
    print('SYNTHETIC_FIXTURE_ONLY',d)
if __name__=='__main__':main()
