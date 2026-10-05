from pathlib import Path
from board import Board

OUT = Path(__file__).resolve().parents[2] / 'static/img/dist'


def fedavg():
    b = Board(1120, 570, 'FedAvg: specify whose examples count', 'Lecture arithmetic, plus an explicit privacy boundary')
    a = b.card(40, 120, 300, 150, 'Client 1', ['100 examples', 'model 0.4', 'contribution 0.1'], 'blue', size=17)
    c = b.card(780, 120, 300, 150, 'Client 2', ['300 examples', 'model 0.8', 'contribution 0.6'], 'orange', size=17)
    m = b.card(400, 130, 320, 130, 'Aggregate', ['(40 + 240) / 400', '= 0.700000'], 'green', size=18)
    b.arrow(a.right(),m.left())
    b.arrow(c.left(),m.right())
    b.card(40, 330, 500, 180, 'What stays local', ['Raw training records stay with each client.', 'Updates still contain information about data.', 'FedAvg alone is not a privacy guarantee.'], 'purple', size=15)
    b.card(580, 330, 500, 180, 'What protection must specify', ['Secure aggregation: hide individual inputs.', 'Differential privacy: bound information release.', 'Neither is established by averaging alone.'], 'teal', size=15)
    return b


def noniid():
    b = Board(1120, 620, 'Heterogeneity makes local steps matter', 'Synthetic clients: h = [1, 4], minima [-1, 1], learning rate 0.1, 24 local steps')
    b.table(55, 125, [140, 180, 220, 230, 240], [['period', 'rounds', 'global weight', 'client gap', 'excess loss'], ['1', '24', '0.599398', '0.320241', '0.000000'], ['4', '6', '0.431989', '0.988154', '0.035284'], ['8', '3', '0.263435', '1.448040', '0.141595']], 'blue', size=15, row_h=47)
    b.card(40, 370, 500, 160, 'Central optimum = 0.600000', ['One local step matches weighted central GD.', 'Many local steps follow different curvatures.', 'Zero target gap starts exactly at the optimum.'], 'green', size=15)
    b.card(580, 370, 500, 160, 'Illustrative decreasing schedule', ['4,4,4,2,2,2,1,1,1,1,1,1', '24 steps, 12 communication rounds', 'excess loss 0.000285; modelled time 720 ms'], 'orange', size=15)
    b.text(560, 580, 'The schedule illustrates adaptation; it is not an implementation of AdaComm.', size=16, color='grey')
    return b


def bank():
    b = Board(1120, 660, 'Distributed ML: six quantities to reconstruct', 'Original revision board; assumptions belong beside each answer')
    entries=[('Amdahl', ['p=0.9, workers=10', 'speedup 5.263158', 'serial ceiling 10'], 'blue'), ('Global batch', ['8 workers × 32 examples', '= 256', 'check gradient normalisation'], 'green'), ('Pipeline bubble', ['4 stages, 8 micro-batches', '3 / 11 = 0.272727', 'ideal equal-time schedule'], 'orange'), ('Ring payload', ['4 workers', 'sent = 1.500 model copies', 'received is separate'], 'purple'), ('Quantisation', ['32-bit / 8-bit = 4', 'ideal payload ratio', 'include scales and packing'], 'teal'), ('FedAvg', ['100 × 0.4 + 300 × 0.8', 'divide by 400 = 0.7', 'sample-weighted objective'], 'red')]
    for i,(title,lines,col) in enumerate(entries):
        b.card(35+(i%3)*365, 115+(i//3)*245, 330, 205, title, lines, col, size=16)
    b.text(560, 625, '29 unique source questions; 15 exact repeats removed from the comprehensive bank.', size=15, color='grey')
    return b


def paper_ps():
    b = Board(1120, 740, 'Q1 redraw: workers and parameter shards', 'Original conceptual redraw of the architecture requested by the question; not a scan reproduction')
    workers=[b.card(40,135+i*125,260,85,'Worker '+str(i),['pull model → compute → push'], 'blue', size=13) for i in range(3)]
    servers=[b.card(750,180+i*190,320,120,'Parameter shard '+str(i),['owns a slice of the weights','reduce gradients and update'], 'orange', size=15) for i in range(2)]
    for w in workers:
        for server in servers: b.arrow(w.right(),server.left(),color='grey',width=1)
    b.card(40,530,1040,100,'Choose an update rule',['sync: wait for required workers','async: accept versioned updates'], 'green', size=15)
    b.card(40,660,1040,50,'Sharding spreads load; fault tolerance also needs recovery and consistent state.',[], 'purple', size=16)
    return b


def paper_pipeline():
    b=Board(1120,620,'Q2 redraw: fill and drain are still real','Original forward-only schedule illustration: four stages, eight micro-batches')
    for stage in range(4):
        b.text(90,165+stage*75,'stage '+str(stage),size=14,color='blue')
        for tick in range(11):
            busy=stage<=tick<stage+8
            b.card(175+tick*80,130+stage*75,70,55,str(tick-stage) if busy else 'idle',[], 'green' if busy else 'grey', size=12)
    b.card(35,465,505,105,'32 busy cells / 44 total',['idle fraction 12/44 = 0.272727'], 'orange', size=17)
    b.card(580,465,505,105,'1F1B mainly bounds live activations',['It does not automatically remove fill/drain.','Interleaving adds a communication trade-off.'], 'purple', size=14)
    return b


def paper_reduce():
    b=Board(1120,620,'Q3 redraw: aggregate gradients before updating','Original redraw; the values below are an added illustration, not scan data')
    a=b.card(35,135,290,150,'Rank 0',['1 example','sum gradient 2','start model 1'], 'blue',size=17)
    c=b.card(795,135,290,150,'Rank 1',['3 examples','sum gradient 12','start model 1'], 'orange',size=17)
    m=b.card(390,145,340,135,'All-reduce SUM',['2 + 12 = 14','divide by 4 examples = 3.5'], 'green',size=16)
    b.arrow(a.right(),m.left()); b.arrow(c.left(),m.right())
    b.card(35,360,500,150,'Update on both ranks',['learning rate 0.1','1 - 0.1 × 3.5 = 0.650000'], 'teal',size=17)
    b.card(585,360,500,150,'Wrong normalisation',['mean of local means: (2 + 4)/2 = 3','wrong model = 0.700000'], 'red',size=16)
    b.text(560,570,'Same reduced gradient and optimiser state are required for identical replica updates.',size=15,color='grey')
    return b



def paper_1f1b():
    stages, microbatches = 4, 8
    queues = []
    for stage in range(stages):
        warmup = min(stages-stage-1, microbatches)
        tasks = [('F', j) for j in range(warmup)]
        for j in range(microbatches-warmup):
            tasks.extend([('F', warmup+j), ('B', j)])
        tasks.extend(('B', j) for j in range(microbatches-warmup, microbatches))
        queues.append(tasks)
    completed = set()
    timeline = [[] for stage in range(stages)]
    while any(queues):
        ready = []
        for stage, queue in enumerate(queues):
            if not queue:
                timeline[stage].append('.')
                continue
            kind, mb = queue[0]
            dependencies = []
            if kind == 'F' and stage > 0:
                dependencies.append((stage-1, 'F', mb))
            if kind == 'B':
                dependencies.append((stage, 'F', mb))
                if stage < stages-1:
                    dependencies.append((stage+1, 'B', mb))
            if all(task in completed for task in dependencies):
                ready.append((stage, kind, mb))
                timeline[stage].append(kind+str(mb))
            else:
                timeline[stage].append('.')
        if not ready:
            raise ValueError('deadlocked schedule')
        for stage, kind, mb in ready:
            queues[stage].pop(0)
            completed.add((stage, kind, mb))
    b = Board(1240, 580, 'Q2 redraw: a complete 1F1B flush schedule', 'Equal-time forward/backward tasks; F and B labels carry micro-batch IDs; dots are idle')
    for stage,row in enumerate(timeline):
        b.text(58,167+stage*70,'stage '+str(stage),size=12,color='grey')
        for tick,task in enumerate(row):
            colour = 'blue' if task.startswith('F') else 'green' if task.startswith('B') else 'grey'
            b.card(110+tick*50,130+stage*70,43,48,task,[],colour,size=10)
    b.card(35,440,555,100,'64 busy cells / 88 total: idle fraction 0.272727',['22 ticks, just like the all-forward/all-backward baseline'], 'orange', size=14)
    b.card(650,440,555,100,'Peak live micro-batches: [4, 3, 2, 1]',['The baseline holds 8 per stage before backward starts.'], 'purple', size=14)
    return b

if __name__ == '__main__':
    for name,make in [('federated-learning-fedavg',fedavg),('special-topics-noniid',noniid),('question-bank-revision',bank),('midsem-parameter-server-redraw',paper_ps),('midsem-pipeline-redraw',paper_pipeline),('midsem-allreduce-redraw',paper_reduce),('midsem-1f1b-redraw',paper_1f1b)]:
        print(make().save(OUT/(name+'.svg')))
