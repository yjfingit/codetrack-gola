#!/usr/bin/env python3
"""Read-only loss/ETA dashboard. UI 20 Hz; process/file telemetry at 1 Hz.

Uses the existing JSON-lines log and run artifacts; never imports torch or touches the
trainer. Progress is the latest *logged* update (currently one record per 16 batches).
"""
from __future__ import annotations

import argparse
from collections import deque
import curses
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import threading
import time
import unicodedata
from zoneinfo import ZoneInfo

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'outputs/causal20_coverage_2gpu_20261008'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def estimated_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--'
    minutes = max(0, math.ceil(seconds / 60))
    days, remainder = divmod(minutes, 1440)
    hours, minutes = divmod(remainder, 60)
    return (f'{days}天' if days else '') + (f'{hours}小时' if days or hours else '') + f'{minutes}分'


def number(value):
    return f'{value:.5g}' if isinstance(value, (float, int)) else '--'


def bar(fraction, width=30):
    fill = round(max(0., min(1., fraction)) * width)
    return '[' + '#' * fill + '-' * (width - fill) + ']'


def finite_values(values):
    return [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x)]


def series_summary(values, window=10):
    """Compare equal adjacent windows within one epoch; this is a descriptive trend."""
    values = finite_values(values)
    if not values:
        return {'count': 0, 'label': '等待样本', 'change': None, 'spike': False}
    recent = values[-window:]
    summary = {'count': len(values), 'latest': values[-1], 'recent': statistics.mean(recent),
               'std': statistics.pstdev(recent), 'minimum': min(values), 'maximum': max(values),
               'label': '样本不足', 'change': None, 'spike': False}
    if len(values) >= window * 2:
        previous = statistics.mean(values[-window * 2:-window])
        summary['previous'] = previous
        if abs(previous) > 1e-12:
            change = (summary['recent'] - previous) / abs(previous) * 100
            summary.update(change=change, label='下降' if change < -3 else '上升' if change > 3 else '基本持平')
    if len(values) >= 11:
        reference = values[-21:-1]
        center = statistics.median(reference)
        mad = statistics.median(abs(x - center) for x in reference)
        # Require both a large relative jump and a robust outlier, not ordinary noise.
        summary['spike'] = values[-1] > center + max(6 * 1.4826 * mad, abs(center) * .5, 1e-6)
    return summary


def smooth(values, window=5):
    return [statistics.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def sparkline(values, width=60, bounds=None):
    values = finite_values(values)[-max(1, width):]
    if not values:
        return '等待数据'
    low, high = bounds if bounds is not None else (min(values), max(values))
    levels = '▁▂▃▄▅▆▇█'
    if high <= low:
        return levels[3] * len(values)
    return ''.join(levels[max(0, min(7, round((x - low) / (high - low) * 7)))] for x in values)


def phase_name(epoch):
    return 'spatial' if epoch < 4 else 'causal8' if epoch < 12 else 'causal16'


def fit_text(value, width):
    """Curses columns are cells, not Python characters (Chinese uses two cells)."""
    result, used = [], 0
    for char in value:
        if not char.isprintable():
            char = ' '
        cells = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used + cells > width:
            break
        result.append(char)
        used += cells
    return ''.join(result)


class LogState:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.identity = None
        self.pending = b''
        self.metric = {}
        self.phase = 'Waiting for log'
        self.losses = deque(maxlen=120)
        self.gradients = deque(maxlen=120)
        self.events = deque(maxlen=6)
        self.last_sequence = {}
        self.last_record = None
        self.last_metric = None
        self.samples = deque(maxlen=8)
        self.error = ''
        self.file_mtime = None

    def consume(self, line, now=None, historical=False):
        now = time.monotonic() if now is None else now
        decoder = json.JSONDecoder()
        cursor = 0
        # Simultaneous rank prints can concatenate JSON objects on one line.
        while True:
            start = line.find('{', cursor)
            if start < 0:
                break
            try:
                row, consumed = decoder.raw_decode(line[start:])
            except ValueError:
                cursor = start + 1
                continue
            cursor = start + consumed
            if not isinstance(row, dict):
                continue
            self.last_record = now
            if 'iteration' in row and 'loss' in row:
                epoch = row['epoch']
                if self.metric.get('epoch') != epoch:
                    self.samples.clear()
                    self.losses.clear()
                    self.gradients.clear()
                self.metric = row
                self.phase = 'TRAINING'
                self.last_metric = now
                self.losses.append(row['loss'])
                if 'grad_norm' in row:
                    self.gradients.append(row['grad_norm'])
                if not historical:
                    chunks = max(1, math.ceil(row.get('phase', {}).get('length', 1) / 4))
                    step = row['iteration'] * chunks + row.get('chunk', 0) // 4 + 1
                    if not self.samples or step > self.samples[-1][1]:
                        self.samples.append((now, step))
                if any(not math.isfinite(row.get(k, 0.)) for k in ('loss', 'grad_norm')):
                    self.error = 'NON-FINITE loss/gradient in log'
            elif 'checkpoint' in row:
                self.phase = 'CHECKPOINT SAVED / BETWEEN PHASES'
                self.events.append(f"Saved epoch_{row.get('epoch', 0):02d}")
            elif 'name' in row and 'PR' in row:
                self.phase = 'EVALUATION (validation or final test)'
                self.last_sequence = row
                self.events.append(f"{row['name']}: PR {row['PR'] * 100:.2f}% SR {row['SR'] * 100:.2f}%")
            elif 'resumed_from' in row:
                self.samples.clear()
                self.events.append(f"Rank {row.get('rank')} resumed at round {row.get('start_epoch', 0) + 1}")
            elif 'rank' in row and 'world_size' in row:
                self.events.append(f"Rank {row['rank']}/{row['world_size']} ready")
        if any(token in line for token in ('Traceback (most recent', 'OutOfMemoryError',
                                           'nonfinite loss', 'non-finite gradients',
                                           'DistBackendError', 'ChildFailedError')):
            self.error = line.strip()[:180]
            self.events.append('ERROR: ' + self.error)

    def poll(self):
        try:
            info = self.path.stat()
        except OSError:
            return
        identity = (info.st_dev, info.st_ino)
        self.file_mtime = info.st_mtime
        initial = self.identity is None
        reset = self.identity != identity or info.st_size < self.offset
        if reset:
            if not initial:
                self.metric = {}
                self.losses.clear()
                self.gradients.clear()
                self.samples.clear()
                self.error = ''
                self.phase = 'Waiting for log'
            self.offset = max(0, info.st_size - 2_000_000)
            self.pending = b''
            self.identity = identity
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            if reset and self.offset:
                stream.readline()  # Discard a partial first line in the bounded backfill.
            chunk = stream.read(2_000_000)
            self.offset = stream.tell()
        parts = (self.pending + chunk).split(b'\n')
        self.pending = parts.pop()
        for line in parts:
            self.consume(line.decode('utf-8', errors='replace'), historical=reset)


class Telemetry(threading.Thread):
    def __init__(self, run, gpus=None):
        super().__init__(daemon=True)
        self.run_dir = run
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.snapshot = {}
        self.coverage_plans = {}

    def run_status(self):
        controls = list(self.run_dir.parent.glob(self.run_dir.name + '*control/status.json'))
        controls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return read_json(controls[0]) if controls else {}

    def collect(self):
        status = self.run_status()
        snapshot = {'status': status, 'ranks': [],
                    'process_error': '', 'checkpoints': [], 'wall': time.time()}
        if psutil is not None:
            for process in psutil.process_iter(['pid', 'cmdline']):
                try:
                    args = process.info['cmdline'] or []
                    if not any(Path(arg).name == 'train_final20_causal.py' for arg in args):
                        continue
                    if str(self.run_dir) not in args:
                        continue
                    # DataLoader workers inherit the script argv; rank parents are torchrun.
                    parent = process.parent()
                    parent_args = ' '.join(parent.cmdline()) if parent else ''
                    if 'torch.distributed.run' not in parent_args:
                        continue
                    snapshot['ranks'].append(process.pid)
                except (psutil.Error, OSError):
                    pass
            pid = status.get('workflow_pid')
            snapshot['workflow_alive'] = bool(pid and psutil.pid_exists(pid))
        else:
            snapshot['workflow_alive'] = None
            snapshot['process_error'] = 'Install psutil for rank liveness (logs still available)'
        for checkpoint in sorted(self.run_dir.glob('epoch_*/model.*')):
            if checkpoint.suffix not in ('.safetensors', '.bin'):
                continue
            snapshot['checkpoints'].append((checkpoint.parent.name,
                                            (checkpoint.parent / 'state.pth').is_file()))
        summaries = sorted(self.run_dir.glob('validation_*/summary.json'))
        snapshot['validation'] = read_json(summaries[-1]) if summaries else {}
        snapshot['selection'] = read_json(self.run_dir / 'selection.json')
        audits=sorted((self.run_dir/'coverage').glob('epoch_*/completed.json'))
        snapshot['coverage']=read_json(audits[-1]) if audits else {}
        for length in (1, 8, 16):
            if length not in self.coverage_plans:
                plan = read_json(self.run_dir / 'coverage' / f'planned_length_{length}.json')
                if plan:
                    self.coverage_plans[length] = {key: value for key, value in plan.items()
                                                   if key != 'per_sequence'}
        snapshot['coverage_plans'] = self.coverage_plans.copy()
        # Checkpoint and validation completion times give usable historical phase
        # rates after a UI restart. Validation time is excluded from training rates.
        protocol = read_json(self.run_dir / 'protocol.json')
        budgets = protocol.get('updates_per_epoch', [])
        starts, phase_costs, validation_costs = {}, {}, []
        protocol_file = self.run_dir / 'protocol.json'
        if protocol_file.exists():
            starts[0] = protocol_file.stat().st_mtime
        for epoch in range(len(budgets)):
            checkpoint = self.run_dir / f'epoch_{epoch:02d}' / 'state.pth'
            if not checkpoint.exists():
                break
            finished = checkpoint.stat().st_mtime
            if epoch in starts and budgets[epoch] and finished > starts[epoch]:
                phase_costs.setdefault(phase_name(epoch), []).append((finished - starts[epoch]) / budgets[epoch])
            validation = self.run_dir / f'validation_{epoch:02d}' / 'summary.json'
            next_start = finished
            if validation.exists():
                next_start = max(finished, validation.stat().st_mtime)
                validation_costs.append(next_start - finished)
            starts[epoch + 1] = next_start
        snapshot['epoch_starts'] = starts
        snapshot['phase_rates'] = {phase: 1 / statistics.median(costs) for phase, costs in phase_costs.items()}
        snapshot['validation_seconds'] = statistics.median(validation_costs) if validation_costs else None
        return snapshot

    def run(self):
        while not self.stop_event.is_set():
            tick = time.monotonic()
            try:
                snapshot = self.collect()
                with self.lock:
                    self.snapshot = snapshot
            except Exception as exc:
                with self.lock:
                    self.snapshot = {**self.snapshot, 'process_error': str(exc)[:120]}
            self.stop_event.wait(max(0., 1. - (time.monotonic() - tick)))

    def get(self):
        with self.lock:
            return self.snapshot.copy()


def estimate_times(state, telemetry, protocol, epoch, step, budget, now):
    rates = dict(telemetry.get('phase_rates', {}))
    phase = phase_name(epoch)
    rate, source = rates.get(phase), '历史轮耗时'
    if len(state.samples) >= 2:
        elapsed = state.samples[-1][0] - state.samples[0][0]
        delta = state.samples[-1][1] - state.samples[0][1]
        if elapsed >= 1 and delta > 0:
            rate, source = delta / elapsed, '最近日志窗口'
    if not rate and state.metric and epoch in telemetry.get('epoch_starts', {}):
        elapsed = time.time() - telemetry['epoch_starts'][epoch]
        if elapsed > 10 and step >= 17:
            rate, source = step / elapsed, '本轮已运行均速'
    if state.phase != 'TRAINING' or (state.file_mtime and time.time() - state.file_mtime > 180):
        return None, None, None, None, '等待训练新记录', []
    if not rate or not budget or protocol.get('smoke'):
        return None, None, None, None, '收集速度样本中', []
    rates[phase] = rate
    budgets = protocol.get('updates_per_epoch', [])
    remaining = max(0, budget - step)
    this_round = remaining / rate
    this_phase = (remaining + sum(b for i, b in enumerate(budgets) if i > epoch and phase_name(i) == phase)) / rate
    missing = list(dict.fromkeys(phase_name(i) for i in range(epoch + 1, len(budgets)) if phase_name(i) not in rates))
    total = None if missing else this_round + sum(budgets[i] / rates[phase_name(i)] for i in range(epoch + 1, len(budgets)))
    return rate, this_round, this_phase, total, source, missing


def display_lines(run, state, telemetry, protocol, now, details=False, width=110, refresh=.05):
    status = telemetry.get('status', {})
    protocol = protocol or read_json(run / 'protocol.json')
    metric = state.metric
    epochs = protocol.get('epochs', 20)
    budgets = protocol.get('updates_per_epoch', [])
    checkpoints = telemetry.get('checkpoints', [])
    epoch = metric.get('epoch', 0)
    chunks = max(1, math.ceil(metric.get('phase', {}).get('length', 1) / 4))
    step = metric.get('iteration', 0) * chunks + metric.get('chunk', 0) // 4 + 1 if metric else 0
    budget = budgets[epoch] if epoch < len(budgets) else 0
    completed = max((int(name.split('_')[-1]) + 1 for name, _ in checkpoints), default=0)
    smoke = protocol.get('smoke', False)
    if completed > epoch and budget and not smoke:
        step = budget
    if smoke:
        budget = chunks
    fraction = step / budget if budget else 0.
    global_step = max(sum(budgets[:epoch]) + step, sum(budgets[:completed]))
    total = sum(budgets)
    rate, round_eta, phase_eta, total_eta, rate_source, missing = estimate_times(
        state, telemetry, protocol, epoch, step, budget, now)
    if completed >= epochs and not smoke:
        rate, round_eta, phase_eta, total_eta, rate_source, missing = None, 0., 0., 0., '20 轮训练已完成', []
    stats = series_summary(state.losses)
    grads = finite_values(state.gradients)
    world = protocol.get('world_size')
    rank_count = len(telemetry.get('ranks', []))
    log_age = time.time() - state.file_mtime if state.file_mtime else None
    phase = {'TRAINING': '训练中', 'Waiting for log': '正在启动',
             'CHECKPOINT SAVED / BETWEEN PHASES': '本轮已保存 / 等待下一阶段',
             'EVALUATION (validation or final test)': '验证或测试中'}.get(state.phase, state.phase)
    run_state = status.get('state', 'UNKNOWN')
    if run_state in ('FAILED', 'COMPLETED') or run_state.startswith('STOPPED'):
        phase = {'FAILED': '运行失败', 'COMPLETED': '工作流已完成'}.get(run_state, '已停止')
    alerts, concerns = [], []
    if state.error:
        alerts.append('日志含错误或非有限损失/梯度；按 d 查看')
    if run_state == 'FAILED':
        alerts.append(f"工作流失败，退出码 {status.get('exit_code', '?')}")
    if run_state == 'RUNNING' and telemetry.get('workflow_alive') is False:
        alerts.append('后台工作流进程已消失')
    if run_state == 'RUNNING' and state.phase == 'TRAINING' and world and rank_count != world and not telemetry.get('process_error'):
        alerts.append(f'DDP 进程不完整：{rank_count}/{world}')
    if state.phase == 'TRAINING' and log_age is not None and log_age > 180:
        concerns.append('超过 3 分钟没有新日志，需检查等待原因')
    if stats['spike']:
        concerns.append('最新损失有明显尖峰')
    if stats['change'] is not None and stats['change'] > 10:
        concerns.append('近期均值上升超过 10%，继续观察')
    if len(grads) >= 11 and grads[-1] > 5 * max(statistics.median(grads[-21:-1]), 1e-6):
        concerns.append('梯度较近期中位数突增超过 5 倍')
    if telemetry.get('process_error'):
        concerns.append('部分监测信息不可用；按 d 查看')
    health = '[异常] ' + '；'.join(alerts) if alerts else '[关注] ' + '；'.join(concerns) if concerns else '[观测] 未见数值或进程异常' if stats['count'] and telemetry else '[等待] 正在收集训练与进程数据'
    rank_text = f'{rank_count}/{world or "?"}' if 'ranks' in telemetry else '采集中'
    clock = datetime.now(ZoneInfo('Asia/Shanghai'))
    clock_text = clock.strftime('%m-%d %H:%M:%S') + f'.{clock.microsecond//10000:02d} CST'
    graph_width = max(12, min(70, width - 27))
    progress_width = max(12, min(34, width - 65))
    stage = {'spatial': '单帧定位', 'causal8': '连续 8 帧', 'causal16': '连续 16 帧'}.get(metric.get('phase', {}).get('phase'), '--')
    lines = [f'CodeTrack LIVE  |  {clock_text}  |  界面 {refresh:g}s / 状态 1s',
             f'{health}  |  {phase}  |  DDP {rank_text}',
             '━━ LOSS · 本轮损失走势 ━━']
    progress_lines = ['━━ 剩余时间与训练进度 ━━',
             f'本轮剩余 ≈ {estimated_duration(round_eta)}  |  本阶段剩余 ≈ {estimated_duration(phase_eta)}',
             f"第 {epoch + 1}/{epochs} 轮 · {stage} · 每卡 batch {metric.get('local_batch', '--')}  |  已保存 {len(checkpoints)}/{epochs} 轮",
             f'本轮 {bar(fraction, progress_width)} {100 * min(1., fraction):5.1f}%  {step:,}/{budget:,} 次更新' if budget else '本轮：等待进度记录',
             (f'试跑检查 {len(checkpoints)}/3，不能计为完整训练轮次' if smoke else
              f'全程 {bar(global_step / total if total else 0., progress_width)} {100 * global_step / total if total else 0.:5.1f}%  {global_step:,}/{total:,} 次更新'),
             f'速度 {number(rate)} 更新/s · {rate_source} · 估时不含验证/最终测试']
    if total_eta is not None:
        progress_lines.insert(2, f'20 轮训练总剩余 ≈ {estimated_duration(total_eta)}（随实际速度修正）')
    else:
        waiting = '、'.join({'causal8': '8 帧', 'causal16': '16 帧', 'spatial': '单帧'}.get(p, p) for p in missing)
        progress_lines.insert(2, f'20 轮训练总剩余：待{waiting}阶段测速' if waiting else '20 轮训练总剩余：收集速度样本中')
    if stats['count']:
        delta = f"{stats['change']:+.1f}%" if stats['change'] is not None else '--'
        previous = number(stats.get('previous'))
        lines.append(f"LOSS 最新 {stats['latest']:.4f}  |  近 10 点均值 {stats['recent']:.4f}  |  前 10 点 {previous}  |  {stats['label']} {delta}")
        values = finite_values(state.losses)[-graph_width:]
        bounds = min(values), max(values)
        lines.append('原始  ' + sparkline(values, graph_width, bounds))
        lines.append('平滑  ' + sparkline(smooth(values), graph_width, bounds) + '  (5 点均值)')
        lines.append(f"曲线范围 {bounds[0]:.3f}–{bounds[1]:.3f}  |  近 10 点波动 σ={stats['std']:.3f}  |  保留 {stats['count']} 条记录")
    else:
        lines.extend(['LOSS 最新 --  |  近 10 点均值 --  |  趋势：等待样本', '原始  等待数据', '平滑  等待数据'])
    if grads:
        lines.append(f'梯度（裁剪前）最新 {grads[-1]:.2f} · 近 10 点中位 {statistics.median(grads[-10:]):.2f} · 最大 {max(grads):.2f}')
    else:
        lines.append('梯度：等待日志')
    lines.extend(progress_lines)
    lines.append('━━ 数据覆盖、保存与验证 ━━')
    checkpoint = checkpoints[-1] if checkpoints else None
    lines.append(f"最近保存 {checkpoint[0] if checkpoint else '--'} · 可恢复状态 {'有' if checkpoint and checkpoint[1] else '--'} · 每 4 轮完整验证")
    length = metric.get('phase', {}).get('length', 1)
    plan = telemetry.get('coverage_plans', {}).get(length, {})
    coverage = telemetry.get('coverage', {})
    if plan and details:
        presentations = ''
        if metric and world and metric.get('local_batch') and not smoke:
            padded = math.ceil(plan['windows'] / (world * metric['local_batch'])) * world * metric['local_batch']
            consumed = min(padded, step // chunks * world * metric['local_batch'])
            presentations = f" · 日志窗口 {consumed:,}/{padded:,}（含补齐）"
        lines.append(f"计划 {plan['sequences']} 段 / {plan['search_frames']:,} 搜索帧{presentations}")
    if coverage:
        result = '完整' if coverage.get('complete') and not coverage.get('smoke') else '未完成（试跑）' if coverage.get('smoke') else '未覆盖完整'
        lines.append(f"实际核对 第 {coverage.get('epoch', 0) + 1} 轮：{result} · {coverage.get('consumed_unique_windows'):,}/{coverage.get('planned_windows'):,} 独立窗口")
    else:
        lines.append('实际覆盖：等待首轮结束核对，计划数量不等于已完成数量')
    validation = telemetry.get('validation', {})
    if validation:
        candidate, baseline = validation.get('candidate', {}), validation.get('baseline', {})
        def percent(value):
            return f'{value * 100:.2f}%' if isinstance(value, (int, float)) else '--'
        def delta(key):
            return f"{(candidate[key] - baseline[key]) * 100:+.2f} pp" if key in candidate and key in baseline else '--'
        lines.append(f"验证 第 {validation.get('epoch', 0) + 1} 轮 · PR {percent(candidate.get('PR'))} ({delta('PR')}) · SR {percent(candidate.get('SR'))} ({delta('SR')})")
        lines.append(f"归一化精度 {percent(candidate.get('NPrecision'))} · 错误写入率 {percent(candidate.get('false_write_rate'))}")
    else:
        lines.append('验证：第 4/8/12/16/20 轮后出结果；损失下降还需验证确认效果')
    lines.append(f'日志距今 {duration(log_age)} · 损失约每 16 批记录 · 曲线自动缩放，跨轮重置')
    if details:
        lines.extend(['━━ 详细信息 ━━', str(run),
                      f'后台 {run_state} · 工作流存活 {telemetry.get("workflow_alive")} · rank PID {telemetry.get("ranks", [])}',
                      f'采样 {protocol.get("sampling", "--")}',
                      f'最近序列 {state.last_sequence.get("name", "--")}'])
        if grads:
            lines.append('梯度  ' + sparkline(grads, graph_width) + '  (>1 会裁剪，不等于异常)')
        validation_time = telemetry.get('validation_seconds')
        if validation_time is not None:
            lines.append(f'已完成验证的典型耗时 {duration(validation_time)}，未计入训练剩余时间')
        for error in (state.error, telemetry.get('process_error')):
            if error:
                lines.append('[异常详情] ' + error)
        lines.extend('  ' + event for event in state.events)
    return lines


def line_style(line):
    def color(pair):
        return curses.color_pair(pair) if curses.has_colors() else curses.A_NORMAL
    if line.startswith('CodeTrack'):
        return color(1) | curses.A_BOLD
    if line.startswith('━━'):
        return color(1) | curses.A_BOLD
    if line.startswith('LOSS 最新'):
        return color(3 if '|  上升' in line else 2) | curses.A_BOLD
    if line.startswith('本轮剩余') or line.startswith('20 轮训练总剩余 ≈'):
        return color(2) | curses.A_BOLD
    if line.startswith('[异常'):
        return color(4) | curses.A_BOLD
    if line.startswith('[关注') or ('|  上升' in line):
        return color(3) | curses.A_BOLD
    if line.startswith('[观测]') or line.startswith('平滑'):
        return color(2)
    if line.startswith('本轮 [') or line.startswith('全程 [') or line.startswith('原始'):
        return color(1)
    if line.startswith('日志距今') or line.startswith('20 轮训练总剩余：待'):
        return curses.A_DIM
    return curses.A_NORMAL


def curses_main(screen, run, state, telemetry, protocol, refresh, details=False):
    curses.set_escdelay(25)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        for pair, foreground in enumerate((curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED), 1):
            curses.init_pair(pair, foreground, background)
    screen.nodelay(True)
    screen.keypad(True)
    previous, scroll = [], 0
    while True:
        tick = time.monotonic()
        state.poll()
        height, width = screen.getmaxyx()
        lines = display_lines(run, state, telemetry.get(), protocol, tick, details, width, refresh)
        available = max(0, height - 2)
        scroll = min(scroll, max(0, len(lines) - 1 - available))
        footer = 'q/Esc 退出面板 · d 详情 · ↑↓/PgUp/PgDn 滚动 · 训练保持运行'
        if len(lines) > height - 1:
            footer += f' · {scroll + 1}–{min(len(lines) - 1, scroll + available)}/{len(lines) - 1}'
        visible = [lines[0]] + lines[1 + scroll:1 + scroll + available]
        visible += [''] * max(0, height - 1 - len(visible))
        visible.append(footer)
        current = []
        for y, line in enumerate(visible[:height]):
            text = fit_text(line, max(0, width - 1))
            style = line_style(line) if y < height - 1 else curses.A_DIM
            current.append((text, style))
            if y >= len(previous) or previous[y] != (text, style):
                try:
                    screen.move(y, 0)
                    screen.clrtoeol()
                    screen.addstr(y, 0, text, style)
                except curses.error:
                    pass
        previous = current
        screen.noutrefresh()
        curses.doupdate()
        key = screen.getch()
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('d'), ord('D')):
            details = not details
        elif key == curses.KEY_DOWN:
            scroll += 1
        elif key == curses.KEY_UP:
            scroll = max(0, scroll - 1)
        elif key == curses.KEY_NPAGE:
            scroll += max(1, available - 2)
        elif key == curses.KEY_PPAGE:
            scroll = max(0, scroll - max(1, available - 2))
        elif key == curses.KEY_HOME:
            scroll = 0
        elif key == curses.KEY_END:
            scroll = max(0, len(lines) - 1 - available)
        elif key == curses.KEY_RESIZE:
            previous = []
            screen.erase()
        telemetry.stop_event.wait(max(0., refresh - (time.monotonic() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--gpus', help=argparse.SUPPRESS)  # Compatibility; no GPU polling.
    parser.add_argument('--refresh', type=float, default=.05)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--details', action='store_true', help='show paths, processes and recent events')
    args = parser.parse_args()
    if args.refresh < .05 or not math.isfinite(args.refresh):
        parser.error('--refresh must be finite and >= 0.05 seconds')
    run = args.run.resolve()
    state = LogState(Path(str(run) + '.log'))
    telemetry = Telemetry(run)
    protocol = read_json(run / 'protocol.json')
    if args.once:
        state.poll()
        print('\n'.join(display_lines(run, state, telemetry.collect(), protocol, time.monotonic(), args.details, refresh=args.refresh)))
        return
    telemetry.start()
    try:
        curses.wrapper(curses_main, run, state, telemetry, protocol, args.refresh, args.details)
    except KeyboardInterrupt:
        pass
    finally:
        telemetry.stop_event.set()


if __name__ == '__main__':
    main()
