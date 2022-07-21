from unit_tests.model_test_utils import extract_model_layers
from dl_code.my_resnet import MyResNet18


def test_my_resnet():
    """
    Tests the transforms using output from disk
    """
    this_res_net = MyResNet18()

    (
        _,
        output_dim,
        _,
        num_params_grad,
        num_params_nograd,
    ) = extract_model_layers(this_res_net)

    print(f'output_dim = {output_dim}\n'
          f'num_params_grad = {num_params_grad}\n'
          f'num_params_nograd = {num_params_nograd}')

    assert output_dim == 2 #15
    assert num_params_grad < 10000
    assert num_params_nograd > 1e7
