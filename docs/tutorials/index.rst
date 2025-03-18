.. DynGPT documentation master file, created by
   sphinx-quickstart on Tue Jan 14 16:22:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Examples
---------

The DynGPT framework is capable of efficiently solving the stationary distribution of state transition networks (STNs)
and inferring static data to obtain the parameters of their underlying STNs. 
The modeling process of STN involves the following steps:

(1) Define the state vector  :math:`s\left( t \right)=\left( {{s}_{1}},\ldots ,{{s}_{M}} \right).`

(2) Identify the set of events :math:`\mathcal{R}=\left\{\left. {{\mathcal{R}}_{1}},\ldots ,{{\mathcal{R}}_{K}} \right\} \right.,` specifying delayed events as needed.

(3) Assign state changes per event :math:`{{\Delta }^{\left(k\right)}}=\left(\Delta _{1}^{\left( k \right)},\ldots ,\Delta _{M}^{\left( k \right)} \right)` to each event.

(4) Define the propensity functions :math:`{{\lambda }_{k}}\left(s;{{\theta }_{k}} \right)` governing the rates of stochastic events.

Under this framework, the system’s state :math:`s\left(t \right)` at any time :math:`t>0` can be written as

.. math::
   s\left( t \right)=s\left( 0 \right)+\sum\nolimits_{k=1}^{K}{{{\Delta }^{\left( k \right)}}{{R}_{k}}\left( t \right)},

where :math:`{{R}_{k}}\left( t \right)` are counting processes that depend on the occurrence propensity and count occurrences of the :math:`k\text{-th}` event :math:`{{\mathcal{R}}_{k}}` up to time :math:`t.`
We demonstrate the application of DynGPT in solving and inferring STNs using the following example:





.. raw:: html

   <table>
      <tr>
         <td style="text-align:center; padding: 20px;">
            <a href="example1/tutorial2.html">
               <img src="../_static/image/Figure2.jpg" alt="Image 1" width="300"/>
               <br/>Gene expression model
            </a>
         </td>
         <td style="text-align:center; padding: 20px;">
            <a href="example2/tutorial3.html">
               <img src="../_static/image/Figure3.jpg" alt="Image 2" width="300"/>
               <br/>Epidemic model
            </a>
         </td>
      </tr>
      <tr>
         <td style="text-align:center; padding: 20px;">
            <a href="example3/tutorial4.html">
               <img src="../_static/image/Figure4.jpg" alt="Image 3" width="300"/>
               <br/>Signaling cascade model
            </a>
         </td>
         <td style="text-align:center; padding: 20px;">
            <a href="example4/tutorial5.html">
               <img src="../_static/image/Figure5.jpg"alt="Image 4" width="300"/>
               <br/>Non-Markovian RNA splicing model
            </a>
         </td>
      </tr>
   </table>






.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   example1/index
   example2/index
   example3/index
   example4/index