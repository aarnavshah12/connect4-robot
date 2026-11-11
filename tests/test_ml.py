"""ML boundary checks: coordinates, unknown scans, legal moves, and training.

Synthetic images below verify the pipeline, not real-camera accuracy.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import board as B
from game import Game, load_config
from ml.cells import (CLASS_NAMES, IMAGE_SIZE, CellClassifier, CellReader,
                      cell_crops, image_tensor, observation)
from ml.generate_games import generate
from ml.mcts import search
from ml.player import PolicyValueNet, TorchPlayer
from ml.positions import encode, position_key
from ml.prepare_board_cells import _grid
from ml.prepare_cells import prepare
from ml.train_player import train as train_player
from ml.train_selfplay import train as train_selfplay
from ml.train_vision import train as train_vision
from tests.test_game import FakeArm, NullOverlay, RecordingSay, ScriptedVision, with_piece
from tests.test_vision import SyntheticCamera
from vision import Vision

CAL = B.Calibration(np.eye(3), 100, 100)


def save_cells(path, label=0):
    model = CellClassifier()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[label] = 20
    torch.save({"format_version": 1, "kind": "cells", "classes": list(CLASS_NAMES),
                "image_size": IMAGE_SIZE, "state_dict": model.state_dict()}, path)


def test_rgb_and_bottom_first_cell_coordinates():
    frame = np.zeros((600, 700, 3), dtype=np.uint8)
    frame[500:600, :100] = (0, 0, 255)
    cells = cell_crops(frame, CAL)
    assert len(cells) == 42
    assert image_tensor(cells[0])[0].mean() == 1  # red is RGB channel 0
    assert image_tensor(cells[35]).sum() == 0   # row 5 is at the image top
    probabilities = np.eye(4)[np.array([1] + [0] * 41)]
    parsed = B.parse_detections(observation(probabilities, CAL, 0.9), CAL)
    assert parsed[0][0] == B.ROBOT and sum(map(sum, parsed)) == B.ROBOT


def test_unknown_and_confidently_empty_are_different():
    empty = np.eye(4)[np.zeros(42, dtype=int)]
    assert observation(empty, CAL, 0.9) == []
    assert B.parse_detections([], CAL) == B.empty_board()
    assert B.parse_detections(None, CAL) is None
    empty[4] = [0, 0, 0, 1]
    assert observation(empty, CAL, 0.9) is None
    empty[4] = [0.4, 0.3, 0.3, 0]
    assert observation(empty, CAL, 0.9) is None


def test_player_masks_full_columns_and_preserves_board(tmp_path):
    model = PolicyValueNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy.bias[0] = 100  # illegal column is most tempting
        model.policy.bias[3] = 5
    path = tmp_path / "player.pt"
    torch.save({"format_version": 1, "kind": "player", "state_dict": model.state_dict(),
                "encoding": "current-opponent-bottom-first"}, path)
    player = TorchPlayer(path)
    board = B.empty_board()
    for row in range(6):
        board[row][0] = 1 + row % 2
    before = [row[:] for row in board]
    considered = []
    col, value = player(board, 2, considered)
    assert col == 3 and value == 0
    assert board == before
    assert 0 not in dict(considered)
    assert sum(p for _, p in considered) == pytest.approx(1)
    with pytest.raises(ValueError, match="full"):
        player([[1] * 7 for _ in range(6)], 1)


def test_optional_tactical_guard_uses_rules_without_minimax(tmp_path):
    model = PolicyValueNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy.bias[6] = 10  # policy prefers a tactically wrong move
    checkpoint = tmp_path / "player.pt"
    torch.save({"format_version": 1, "kind": "player", "state_dict": model.state_dict(),
                "encoding": "current-opponent-bottom-first"}, checkpoint)
    pure, guarded = TorchPlayer(checkpoint), TorchPlayer(checkpoint, tactical_guard=True)
    winning = B.empty_board()
    for col in range(3):
        winning[0][col] = B.ROBOT
    assert pure(winning, B.ROBOT)[0] == 6
    assert guarded(winning, B.ROBOT)[0] == 3
    threatened = B.empty_board()
    for col in range(3):
        threatened[0][col] = B.HUMAN
    assert guarded(threatened, B.ROBOT)[0] == 3


def test_neural_search_finds_a_win_without_changing_board():
    model = PolicyValueNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    board = B.empty_board()
    for col in range(3):
        board[0][col] = B.ROBOT
    before = [row[:] for row in board]
    col, visits, value = search(model, board, B.ROBOT, simulations=16)
    assert col == 3
    assert visits.sum() == pytest.approx(1)
    assert value > 0
    assert board == before


def test_selfplay_training_writes_a_loadable_checkpoint(tmp_path):
    initial = PolicyValueNet()
    start = tmp_path / "start.pt"
    torch.save({"format_version": 1, "kind": "player", "state_dict": initial.state_dict(),
                "encoding": "current-opponent-bottom-first"}, start)
    output = tmp_path / "selfplay"
    history = train_selfplay(start, output, iterations=1, games_per_iteration=8,
                             ppo_epochs=1, opponent_depth=2)
    assert history[0]["games"] == 8 and history[0]["positions"] > 0
    player = TorchPlayer(output / "last.pt", search_simulations=8)
    assert B.landing_row(B.empty_board(), player(B.empty_board(), B.ROBOT)[0]) is not None


def test_position_encoding_and_mirror_groups():
    board = B.empty_board()
    board[0][0], board[0][1] = 1, 2
    planes = encode(board, 2)
    assert planes[0, 0, 1] == 1 and planes[1, 0, 0] == 1
    assert planes.sum() == 2
    assert position_key(board, 2) == position_key([row[::-1] for row in board], 2)


def test_board_cell_geometry_recovers_shuffled_perspective_grid():
    source = np.float32([((col + 0.5) * 100, (row + 0.5) * 100)
                         for row in range(6) for col in range(7)])
    corner_src = np.float32([[50, 50], [650, 50], [650, 550], [50, 550]])
    corner_dst = np.float32([[120, 45], [810, 90], [780, 680], [80, 630]])
    transform = cv2.getPerspectiveTransform(corner_src, corner_dst)
    distorted = cv2.perspectiveTransform(source[:, None, :], transform)[:, 0, :]
    order = np.random.default_rng(42).permutation(42)
    annotations = [{"bbox": [float(distorted[i, 0] - 20), float(distorted[i, 1] - 20),
                             40.0, 40.0]} for i in order]
    recovered = _grid(annotations)
    assert recovered is not None
    homography, cells = recovered
    assert cells == [divmod(int(i), 7) for i in order]
    mapped = cv2.perspectiveTransform(distorted[:, None, :], homography)[:, 0, :]
    assert np.max(abs(mapped - source)) < 1


def test_game_counts_distinct_results_and_resets_on_unknown():
    class HeldVision:
        backend = "torch"
        fid, dets = 1, []

        def detections(self):
            return self.fid, self.dets, 10.0

    vision = HeldVision()
    game = Game(vision, FakeArm(), NullOverlay(), RecordingSay(), CAL)
    for _ in range(8):
        assert game._poll_stable() is None
    assert game.debounce._count == 1
    vision.fid, vision.dets = 2, None
    assert game._poll_stable() is None and game.debounce._count == 0
    vision.dets = []
    for fid in range(3, 7):
        vision.fid = fid
        assert game._poll_stable() is None
    vision.fid = 7
    assert game._poll_stable() == B.empty_board()


def test_torch_config_needs_no_roboflow_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("vision_backend: torch\nplayer_backend: torch\n")
    assert load_config(path)["vision_backend"] == "torch"


def test_neural_player_full_turn_uses_learned_scale_and_perspective():
    class LearnedMoves:
        score_kind = "learned_value"

        def __call__(self, board, player, considered=None):
            if considered is not None:
                considered.append((4, 0.7))
            return 4, 0.8 if player == B.ROBOT else 0.6

    initial = B.empty_board()
    human = with_piece(initial, 3, B.HUMAN)
    vision, arm = ScriptedVision([initial, human]), FakeArm()
    overlay, say = NullOverlay(), RecordingSay()
    game = Game(vision, arm, overlay, say, CAL, sleep=lambda _: None, move_fn=LearnedMoves())
    arm.on_drop = lambda col: vision.frames.extend([with_piece(human, col, B.ROBOT)] * 10)
    assert game.play_turn() == "ok"
    assert arm.dropped == [4] and "blunder" in say.triggers
    assert "foreshadow" not in say.triggers
    assert game._talk_ctx()["eval_kind"] == "learned_value"
    assert any(p.get("eval_kind") == "learned_value" for p in overlay.published)


def test_torch_worker_to_board_without_hardware(tmp_path):
    checkpoint = tmp_path / "cells.pt"
    save_cells(checkpoint, label=1)
    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({"homography": np.eye(3).tolist(), "cell_w": 8, "cell_h": 8,
                                "crop": [0, 0, 64, 48], "frame_size": [64, 48]}))
    vision = Vision({"vision_backend": "torch", "vision_checkpoint": str(checkpoint),
                     "vision_device": "cpu"}, calib_path=calib, camera=SyntheticCamera())
    vision.start()
    try:
        deadline = time.monotonic() + 20
        dets = None
        while time.monotonic() < deadline:
            _, dets, _ = vision.detections()
            if dets:
                break
            time.sleep(0.05)
        assert dets is not None and len(dets) == 42
        parsed = B.parse_detections(dets, B.Calibration.load(calib))
        assert parsed == [[B.ROBOT] * 7 for _ in range(6)]
    finally:
        vision.stop()


def make_coco(root, duplicate=False):
    for split, intensity in (("train", 40), ("valid", 80), ("test", 120)):
        folder = root / split
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "frame.png"), np.full((30, 90, 3), 40 if duplicate else intensity, np.uint8))
        (folder / "_annotations.coco.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "frame.png"}],
            "categories": [{"id": i, "name": name} for i, name in
                           enumerate(("No Piece", "Red Piece", "Yellow Piece"))],
            "annotations": [{"id": i, "image_id": 1, "category_id": i,
                             "bbox": [i * 30, 0, 30, 30]} for i in range(3)]}))


def test_coco_conversion_and_duplicate_split_rejection(tmp_path):
    make_coco(tmp_path / "raw")
    counts = prepare(tmp_path / "raw", tmp_path / "cells")
    assert counts["train/red"] == 1 and counts["test/empty"] == 1
    assert len(json.loads((tmp_path / "cells/manifest.json").read_text())) == 9
    make_coco(tmp_path / "duplicate", duplicate=True)
    with pytest.raises(ValueError, match="identical source image"):
        prepare(tmp_path / "duplicate", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_real_vision_training_loop_saves_loadable_checkpoint(tmp_path):
    make_coco(tmp_path / "raw")
    prepare(tmp_path / "raw", tmp_path / "cells")
    history = train_vision(tmp_path / "cells", tmp_path / "vision", epochs=1, device="cpu")
    assert np.isfinite(history[0]["train_loss"])
    reader = CellReader(tmp_path / "vision/best.pt", CAL, threshold=0.01)
    assert reader.model.training is False
    assert reader.model(image_tensor(np.zeros((64, 64, 3), np.uint8)).unsqueeze(0)).shape == (1, 4)


def test_player_training_and_leakage_free_splits(tmp_path):
    path = tmp_path / "games.npz"
    generate(path, games=12, teacher_depth=2, seed=5)
    keys, game_ids = [], []
    with np.load(path, allow_pickle=False) as data:
        for split in ("train", "val", "test"):
            boards, players = data[f"{split}_boards"], data[f"{split}_players"]
            keys.append({position_key(b, p) for b, p in zip(boards, players)})
            game_ids.append(set(data[f"{split}_game_ids"]))
            assert np.isfinite(data[f"{split}_values"]).all()
            assert np.max(abs(data[f"{split}_values"])) <= 1
        for i in range(3):
            for j in range(i + 1, 3):
                assert not keys[i] & keys[j]
                assert not game_ids[i] & game_ids[j]
    history = train_player(path, tmp_path / "player", epochs=1, device="cpu")
    assert np.isfinite(history[0]["train_loss"])
    col, value = TorchPlayer(tmp_path / "player/best.pt")(B.empty_board(), 1)
    assert 0 <= col < 7 and -1 <= value <= 1
