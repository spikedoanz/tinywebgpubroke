```
pip install -r requirements.txt
python tiny_meshnet.py
```

using default backend (metal in my case), and webgpu produces different results.
```
python tiny_meshnet.py
WEBGPU=1 tiny_meshnet.py
```


produces different nifies (visualized in brainchop.org)

![webgpu](webgpu.png)
![metal](metal.png)


see also

```
metal.nii.gz
webgpu.nii.gz
```

using the latest master branch from github gives a different error
```
(.venv) repos/tinywebgpubroke § WEBGPU=1 python tiny_meshnet.py
ram used:  0.00 GB, model.75.weight                                   : 100%|████████████████████████████████████████████████████| 26/26 [00:00<00:00, 418.66it/s]
loaded weights in  62.27 ms, 0.00 GB loaded at 0.01 GB/s
Loaded weights from model.pth...
Loading t1_crop.nii.gz...
Volume shape: (256, 256, 256)
Preprocessing volume...
Running inference...
OUT OF BOUNDS ACCESS in INDEX 0 - 119 not in 0 - 15. idx.src[1].render()='(lidx0+(gidx1*3))'
   0 Ops.NAME            : dtypes.void                    []                               E_5_262144_3_16_4
   1 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(251658240)    []                               0
   2 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(251658240)    []                               1
   3 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(15)           []                               2
   4 Ops.DEFINE_GLOBAL   : dtypes.float.ptr(15)           []                               3
   5 Ops.SPECIAL         : dtypes.int                     []                               ('gidx0', 32768)
   6 Ops.SPECIAL         : dtypes.int                     []                               ('gidx1', 40)
   7 Ops.SPECIAL         : dtypes.int                     []                               ('lidx0', 3)
   8 Ops.SPECIAL         : dtypes.int                     []                               ('lidx1', 16)
   9 Ops.CONST           : dtypes.float                   []                               -2.302208198144325
  10 Ops.CONST           : dtypes.float                   []                               0.044715
  11 Ops.CONST           : dtypes.int                     []                               1
  12 Ops.CONST           : dtypes.float                   []                               1.0
  13 Ops.CONST           : dtypes.int                     []                               2
  14 Ops.CONST           : dtypes.uint                    []                               2
  15 Ops.CONST           : dtypes.int                     []                               3
  16 Ops.CONST           : dtypes.uint                    []                               6
  17 Ops.CONST           : dtypes.uint                    []                               24
  18 Ops.CONST           : dtypes.int                     []                               50331648
  19 Ops.NOOP            : dtypes.int                     [5]                              None
  20 Ops.NOOP            : dtypes.int                     [7]                              None
  21 Ops.NOOP            : dtypes.int                     [8]                              None
  22 Ops.BITCAST         : dtypes.uint                    [19]                             None
  23 Ops.BITCAST         : dtypes.uint                    [20]                             None
  24 Ops.BITCAST         : dtypes.uint                    [21]                             None
  25 Ops.MUL             : dtypes.int                     [6, '3']                         None
  26 Ops.MUL             : dtypes.int                     [6, '50331648']                  None
  27 Ops.SHL             : dtypes.uint                    [22, '6']                        None
  28 Ops.NOOP            : dtypes.uint                    [27]                             None
  29 Ops.BITCAST         : dtypes.int                     [28]                             None
  30 Ops.SHL             : dtypes.uint                    [23, '24']                       None
  31 Ops.NOOP            : dtypes.uint                    [30]                             None
  32 Ops.BITCAST         : dtypes.int                     [31]                             None
  33 Ops.SHL             : dtypes.uint                    [24, '2']                        None
  34 Ops.NOOP            : dtypes.uint                    [33]                             None
  35 Ops.BITCAST         : dtypes.int                     [34]                             None
  36 Ops.ADD             : dtypes.int                     [7, 25]                          None
  37 Ops.INDEX           : dtypes.float.ptr(15)           [3, 36]                          None
  38 Ops.LOAD            : dtypes.float                   [37]                             None
  39 Ops.INDEX           : dtypes.float.ptr(15)           [4, 36]                          None
  40 Ops.LOAD            : dtypes.float                   [39]                             None
  41 Ops.ADD             : dtypes.int                     [29, 26]                         None
  42 Ops.ADD             : dtypes.int                     [32, 41]                         None
  43 Ops.ADD             : dtypes.int                     [35, 42]                         None
  44 Ops.INDEX           : dtypes.float.ptr(251658240)    [2, 43]                          None
  45 Ops.LOAD            : dtypes.float                   [44]                             None
  46 Ops.ADD             : dtypes.int                     [43, '1']                        None
  47 Ops.INDEX           : dtypes.float.ptr(251658240)    [2, 46]                          None
  48 Ops.LOAD            : dtypes.float                   [47]                             None
  49 Ops.ADD             : dtypes.int                     [43, '2']                        None
  50 Ops.INDEX           : dtypes.float.ptr(251658240)    [2, 49]                          None
  51 Ops.LOAD            : dtypes.float                   [50]                             None
  52 Ops.ADD             : dtypes.int                     [43, '3']                        None
  53 Ops.INDEX           : dtypes.float.ptr(251658240)    [2, 52]                          None
  54 Ops.LOAD            : dtypes.float                   [53]                             None
  55 Ops.INDEX           : dtypes.float.ptr(251658240)    [1, 43]                          None
  56 Ops.INDEX           : dtypes.float.ptr(251658240)    [1, 46]                          None
  57 Ops.INDEX           : dtypes.float.ptr(251658240)    [1, 49]                          None
  58 Ops.INDEX           : dtypes.float.ptr(251658240)    [1, 52]                          None
  59 Ops.SUB             : dtypes.float                   [45, 38]                         None
  60 Ops.MUL             : dtypes.float                   [59, 40]                         None
  61 Ops.MUL             : dtypes.float                   [60, 60]                         None
  62 Ops.MUL             : dtypes.float                   [61, 60]                         None
  63 Ops.MUL             : dtypes.float                   ['0.044715', 62]                 None
  64 Ops.ADD             : dtypes.float                   [60, 63]                         None
  65 Ops.MUL             : dtypes.float                   [64, '-2.302208198144325']       None
  66 Ops.EXP2            : dtypes.float                   [65]                             None
  67 Ops.ADD             : dtypes.float                   ['1.0', 66]                      None
  68 Ops.RECIP           : dtypes.float                   [67]                             None
  69 Ops.MUL             : dtypes.float                   [68, 60]                         None
  70 Ops.STORE           : dtypes.void                    [55, 69]                         None
  71 Ops.SUB             : dtypes.float                   [48, 38]                         None
  72 Ops.MUL             : dtypes.float                   [71, 40]                         None
  73 Ops.MUL             : dtypes.float                   [72, 72]                         None
  74 Ops.MUL             : dtypes.float                   [73, 72]                         None
  75 Ops.MUL             : dtypes.float                   ['0.044715', 74]                 None
  76 Ops.ADD             : dtypes.float                   [72, 75]                         None
  77 Ops.MUL             : dtypes.float                   [76, '-2.302208198144325']       None
  78 Ops.EXP2            : dtypes.float                   [77]                             None
  79 Ops.ADD             : dtypes.float                   ['1.0', 78]                      None
  80 Ops.RECIP           : dtypes.float                   [79]                             None
  81 Ops.MUL             : dtypes.float                   [80, 72]                         None
  82 Ops.STORE           : dtypes.void                    [56, 81]                         None
  83 Ops.SUB             : dtypes.float                   [51, 38]                         None
  84 Ops.MUL             : dtypes.float                   [83, 40]                         None
  85 Ops.MUL             : dtypes.float                   [84, 84]                         None
  86 Ops.MUL             : dtypes.float                   [85, 84]                         None
  87 Ops.MUL             : dtypes.float                   ['0.044715', 86]                 None
  88 Ops.ADD             : dtypes.float                   [84, 87]                         None
  89 Ops.MUL             : dtypes.float                   [88, '-2.302208198144325']       None
  90 Ops.EXP2            : dtypes.float                   [89]                             None
  91 Ops.ADD             : dtypes.float                   ['1.0', 90]                      None
  92 Ops.RECIP           : dtypes.float                   [91]                             None
  93 Ops.MUL             : dtypes.float                   [92, 84]                         None
  94 Ops.STORE           : dtypes.void                    [57, 93]                         None
  95 Ops.SUB             : dtypes.float                   [54, 38]                         None
  96 Ops.MUL             : dtypes.float                   [95, 40]                         None
  97 Ops.MUL             : dtypes.float                   [96, 96]                         None
  98 Ops.MUL             : dtypes.float                   [97, 96]                         None
  99 Ops.MUL             : dtypes.float                   ['0.044715', 98]                 None
 100 Ops.ADD             : dtypes.float                   [96, 99]                         None
 101 Ops.MUL             : dtypes.float                   [100, '-2.302208198144325']      None
 102 Ops.EXP2            : dtypes.float                   [101]                            None
 103 Ops.ADD             : dtypes.float                   ['1.0', 102]                     None
 104 Ops.RECIP           : dtypes.float                   [103]                            None
 105 Ops.MUL             : dtypes.float                   [104, 96]                        None
 106 Ops.STORE           : dtypes.void                    [58, 105]                        None
Traceback (most recent call last):
  File "/Users/mdoan4/repos/tinywebgpubroke/tiny_meshnet.py", line 167, in <module>
    run_inference(model, nifti_path, output_path)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tiny_meshnet.py", line 134, in run_inference
    output = model(input_tensor).realize()
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/tensor.py", line 4198, in _wrapper
    ret = fn(*args, **kwargs)
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/tensor.py", line 256, in realize
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/engine/realize.py", line 167, in run_schedule
    for si, ei in lower_schedule(schedule):
                  ~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/engine/realize.py", line 160, in lower_schedule
    raise e
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/engine/realize.py", line 154, in lower_schedule
    try: yield (si, lower_schedule_item(si))
                    ~~~~~~~~~~~~~~~~~~~^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/engine/realize.py", line 149, in lower_schedule_item
    def lower_schedule_item(si:ScheduleItem) -> ExecItem: return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata)
                                                                                                    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/ops.py", line 834, in rewrite
    if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None: return ret
              ~~~^^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/engine/realize.py", line 143, in <lambda>
    (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
                                                            ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/engine/realize.py", line 111, in get_runner
    prg: ProgramSpec = get_kernel(Device[device].renderer, ast).to_program()
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/codegen/kernel.py", line 686, in to_program
    self.linearize(name_override, ast_transform)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/ops.py", line 858, in __wrapper
    ret = func(self, *args, **kwargs)
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/codegen/kernel.py", line 681, in linearize
    self.uops:list[UOp] = linearize_uop(full_graph_rewrite(rewrite_shapetracker_with_index(modified_ast, self.opts), self.opts))
                          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/codegen/linearize.py", line 251, in linearize_uop
    if not skip_check: type_verify(sink.arg.lst)
                       ~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/Users/mdoan4/repos/tinywebgpubroke/tg/tinygrad/spec.py", line 171, in type_verify
    raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[x.op for x in u.src]} {u.arg}")
RuntimeError: UOp verification failed at 37 on Ops.INDEX dtypes.float.ptr(15) 2 [<Ops.DEFINE_GLOBAL: 24>, <Ops.ADD: 54>] None
```
