parameters {
  real y;
  vector[9] z;
}

transformed parameters {
  vector[9] x;

  x = exp(y/2) * z;

}

model {
  y ~ normal(0, 3);
  z ~ std_normal();
}
