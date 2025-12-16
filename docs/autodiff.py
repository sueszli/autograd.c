import ast
import astunparse

import os
from math import exp, cos, sin
from collections import namedtuple
from numbers import Number

import unittest
import jax.numpy as jnp
from jax import grad
import torch


# idea: instead of splitting up types and methods as done below, we could also have
# a single abstract data type and overload primitive operators through __add__, __mul__, etc.
DualNum = namedtuple("DualNum", ["value", "derivative"])


# operators are incomplete and only serve the purpose of our single example function f(x)
# see:
# - https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
# - https://youtu.be/5F6roh4pmJU?si=LW1ZKKvaGdl9shCz&t=555
class DualNumOps:
    @staticmethod
    def custom_exp(inp: DualNum):
        return DualNum(exp(inp.value), exp(inp.value) * inp.derivative)

    @staticmethod
    def custom_cos(inp: DualNum):
        return DualNum(cos(inp.value), -sin(inp.value) * inp.derivative)

    @staticmethod
    def custom_add(inp1: DualNum, inp2: DualNum):
        if not isinstance(inp1, DualNum):
            inp1 = DualNum(inp1, 0.0)
        if not isinstance(inp2, DualNum):
            inp2 = DualNum(inp2, 0.0)

        return DualNum(inp1.value + inp2.value, inp1.derivative + inp2.derivative)

    @staticmethod
    def custom_mul(inp1: DualNum, inp2: DualNum):
        return DualNum(
            inp1.value * inp2.value,
            inp1.derivative * inp2.value + inp2.derivative * inp1.value,
        )

    @staticmethod
    def custom_pow(inp: DualNum, k: Number):
        assert isinstance(k, int), "k must be an integer"
        k_int = int(k)
        return DualNum(inp.value**k, inp.derivative * k * inp.value ** (k_int - 1))


# update the abstract syntax tree
# see: 
# - https://docs.python.org/3/library/ast.html
# - https://greentreesnakes.readthedocs.io/en/latest/index.html
# - https://greentreesnakes.readthedocs.io/en/latest/manipulating.html
def transform(fstr: str) -> str:
    class CustomOpTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = "f_forward_ad"
            node.args.args[0].annotation = ast.Name(id="DualNum", ctx=ast.Load())
            node.returns = ast.Name(id="DualNum", ctx=ast.Load())

            self.generic_visit(node)  # visit children
            return node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                if node.func.id == "exp":
                    node.func.id = "DualNumOps.custom_exp"
                elif node.func.id == "cos":
                    node.func.id = "DualNumOps.custom_cos"

            self.generic_visit(node)  # visit children
            return node

        def visit_BinOp(self, node):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                return node  # both are constants, reached leaf

            if isinstance(node.op, ast.Add):
                node = ast.Call(func=ast.Name(id="DualNumOps.custom_add", ctx=ast.Load()), args=[node.left, node.right], keywords=[])
            elif isinstance(node.op, ast.Mult):
                node = ast.Call(func=ast.Name(id="DualNumOps.custom_mul", ctx=ast.Load()), args=[node.left, node.right], keywords=[])
            elif isinstance(node.op, ast.Pow):
                node = ast.Call(func=ast.Name(id="DualNumOps.custom_pow", ctx=ast.Load()), args=[node.left, node.right], keywords=[])

            self.generic_visit(node)  # visit children
            return node

    tree = ast.parse(fstr)
    transformer = CustomOpTransformer()
    tree = transformer.visit(tree)  # call root

    new_fstr = astunparse.unparse(tree)
    return new_fstr


class TestCases(unittest.TestCase):
    jax_func = lambda x: jnp.exp(x) ** 3 + jnp.cos(x) * x + 10**2
    torch_func = lambda x: torch.exp(x) ** 3 + torch.cos(x) * x + 10**2

    def test_jax_1(self):
        testarg = 2.4
        res_jax = grad(TestCases.jax_func)(testarg)
        res_custom = f_forward_ad(DualNum(testarg, 1.0)).derivative  # type: ignore
        res_match = jnp.allclose(res_jax, res_custom)
        self.assertTrue(res_match)

    def test_jax_2(self):
        testarg = 61.78
        res_jax = grad(TestCases.jax_func)(testarg)
        res_custom = f_forward_ad(DualNum(testarg, 1.0)).derivative  # type: ignore
        res_match = jnp.allclose(res_jax, res_custom)
        self.assertTrue(res_match)

    def test_jax_3(self):
        testarg = 26.42
        res_jax = grad(TestCases.jax_func)(testarg)
        res_custom = f_forward_ad(DualNum(testarg, 1.0)).derivative  # type: ignore
        res_match = jnp.allclose(res_jax, res_custom)
        self.assertTrue(res_match)

    def test_torch_1(self):
        testarg = 2.4
        x = torch.tensor(testarg, requires_grad=True)
        TestCases.torch_func(x).backward()
        result_pytorch = x.grad.item() if x.grad is not None else None
        result_custom = f_forward_ad(DualNum(testarg, 1.0)).derivative  # type: ignore
        torch_match = torch.isclose(torch.tensor(result_pytorch), torch.tensor(result_custom))
        self.assertTrue(torch_match)

    def test_torch_2(self):
        testarg = 61.78
        x = torch.tensor(testarg, requires_grad=True)
        TestCases.torch_func(x).backward()
        result_pytorch = x.grad.item() if x.grad is not None else None
        result_custom = f_forward_ad(DualNum(testarg, 1.0)).derivative  # type: ignore
        torch_match = torch.isclose(torch.tensor(result_pytorch), torch.tensor(result_custom))
        self.assertTrue(torch_match)

    def test_torch_3(self):
        testarg = 26.42
        x = torch.tensor(testarg, requires_grad=True)
        TestCases.torch_func(x).backward()
        result_pytorch = x.grad.item() if x.grad is not None else None
        result_custom = f_forward_ad(DualNum(testarg, 1.0)).derivative  # type: ignore
        torch_match = torch.isclose(torch.tensor(result_pytorch), torch.tensor(result_custom))
        self.assertTrue(torch_match)


f_str = """
def f(x):
    return exp(x)**3 + cos(x) * x + 10**2
"""

# the following is the AST of `f_str` as dumped by `ast.dump(tree)` for reference
"""
Module(
    body=[
        FunctionDef(
            body=[
                Return(
                    value=BinOp(
                        
                        left=BinOp(                                  <------ `exp(x)**3 + cos(x) * x` AS LEFT
                            left=BinOp(                              <------ `exp(x)**3` AS LEFT
                                left=Call(
                                    func=Name(id='exp', ctx=Load()),
                                    name='f',
                                    args=arguments(
                                        posonlyargs=[],
                                        args=[arg(arg='x')],
                                        kwonlyargs=[],
                                        kw_defaults=[],
                                        defaults=[]
                                    ),
                                    args=[Name(id='x', ctx=Load())],
                                    keywords=[]
                                ),
                                op=Pow(),
                                right=Constant(value=3)
                            ),
                            
                            op=Add(),                               <------ LEFT + RIGHT
                            
                            right=BinOp(                            <------ `cos(x) * x` AS RIGHT
                                left=Call(
                                    func=Name(id='cos', ctx=Load()),
                                    args=[Name(id='x', ctx=Load())],
                                    keywords=[]
                                ),
                                op=Mult(),
                                right=Name(id='x', ctx=Load())
                            )
                        ),
                        
                        op=Add(),                                 <------ LEFT + RIGHT
                        
                        right=BinOp(                              <------ `10**2` AS RIGHT
                            left=Constant(value=10),
                            op=Pow(),
                            right=Constant(value=2)
                        )
                    )
                )
            ],
            decorator_list=[],
            type_params=[]
        )
    ],
    type_ignores=[]
)
"""

if __name__ == "__main__":
    # clear screen
    os.system("cls" if os.name == "nt" else "clear")
    os.system("uname -a") if os.name == "posix" else os.system("systeminfo")

    # bring f(x) into local namespace
    exec(f_str)
    assert "f" in locals(), "f is not defined"
    assert f(2) == exp(2) ** 3 + cos(2) * 2 + 10**2  # type: ignore

    # get f_forward_ad(x) from f(x)
    python_str = transform(f_str)
    print(f"\n\nOriginal function:\n\033[92m{f_str}\033[0m\nTransformed function:\033[92m{python_str}\033[0m\n\n")

    # bring f_forward_ad(x) into local namespace
    exec(python_str)
    assert "f_forward_ad" in locals(), "f_forward_ad is not defined"

    # run tests
    unittest.main()
