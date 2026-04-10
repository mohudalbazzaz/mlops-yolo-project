from src.backend.train_model import save_model, MODEL_NAME


def test_save_model(monkeypatch):
    fake_images = {"folder1": [1, 2, 3]}
    monkeypatch.setattr(
        "src.backend.train_model.extract_imgs_from_db", lambda bucket: fake_images
    )

    fake_splits = (
        ["X_train"],
        ["y_train"],
        ["X_val"],
        ["y_val"],
        ["X_test"],
        ["y_test"],
    )
    monkeypatch.setattr(
        "src.backend.train_model.split_data", lambda images: fake_splits
    )

    class DummyModel:
        def save(self, path):
            self.saved_path = path

    dummy_model = DummyModel()
    monkeypatch.setattr(
        "src.backend.train_model.train_and_validate_model",
        lambda *args: (None, dummy_model, None),
    )

    save_model()

    assert dummy_model.saved_path == MODEL_NAME
